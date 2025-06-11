import os
import torch
import numpy as np
import wandb
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, Grayscale, Resize
from torch.utils.data import DataLoader, Dataset
from skimage.metrics import structural_similarity as ssim_metric, peak_signal_noise_ratio as psnr_metric
from Resources.Ultris.Ultris_model import UNet
from Resources.Ultris.Ultris_zernike import generate_zernike_map, generate_psf, apply_psf, device
import torch.nn.functional as F

# Configuraciones generales
n, m = 2, 2
num_epochs = 200
image_dir = "Resources/Images"
output_root = "Resources/resultados_epocas"
os.makedirs(output_root, exist_ok=True)

#resize_to_512 = Resize((512, 512))
#resize_to_512 = Resize((1024, 1024))
resize_to_512 = Resize((256, 256))

# Dataset personalizado
def load_image_tensor(path):
    img = Image.open(path).convert("RGB")
    img = Grayscale()(img)
    img = resize_to_512(img)
    return ToTensor()(img).to(device)

class ImageDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return load_image_tensor(self.paths[idx])

all_image_paths = [os.path.join(image_dir, f"{i:06d}.png") for i in range(10)]
train_paths = all_image_paths[:7]
val_paths = all_image_paths[7:]

train_loader = DataLoader(ImageDataset(train_paths), batch_size=1, shuffle=True)
val_loader = DataLoader(ImageDataset(val_paths), batch_size=1, shuffle=False)

# Recorte centrado
CROP = 20
def crop_center(img):
    return img[CROP:-CROP, CROP:-CROP]

# Función para aplicar PSF
def apply_psf_torch(image_tensor, psf_tensor):
    if psf_tensor.max() == 1.0 and psf_tensor.sum() == 1.0 and torch.count_nonzero(psf_tensor) == 1:
        return image_tensor
    return F.conv2d(image_tensor, psf_tensor, padding="same")

# Lista de severidades a procesar
amplitudes = [0.5, 1.0, 2.0, 3.0]

for amplitud in amplitudes:
    print(f"Entrenando para amplitud {amplitud}")
    wandb.init(project="restauracion-zernike", config={"n": n, "m": m, "amplitud": amplitud, "epochs": num_epochs}, dir="Resources")

    # Crear PSF
    zmap = generate_zernike_map(n, m, amplitude=amplitud)
    psf = generate_psf(zmap)
    psf_tensor = psf.unsqueeze(0).unsqueeze(0).to(device)

    # Crear carpetas por severidad
    output_dir = os.path.join(output_root, f"amplitud_{amplitud:.2f}")
    os.makedirs(output_dir, exist_ok=True)

    # Inicializar modelo
    model = UNet(in_channels=1, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        train_mses, train_ssims, train_psnrs = [], [], []

        for x in train_loader:
            x = x.to(device)
            fx = model(x)
            fx_blur = apply_psf_torch(fx, psf_tensor)

            loss = loss_fn(fx_blur, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            fx_np = crop_center(fx.squeeze().detach().cpu().numpy())
            x_np = crop_center(x.squeeze().detach().cpu().numpy())
            fx_np = np.clip(fx_np, 0, 1)
            x_np = np.clip(x_np, 0, 1)
            train_mses.append(np.mean((fx_np - x_np)**2))
            train_ssims.append(ssim_metric(x_np, fx_np, data_range=1.0))
            train_psnrs.append(psnr_metric(x_np, fx_np, data_range=1.0))
  


        # Evaluación
        model.eval()
        val_losses, val_mses, val_ssims, val_psnrs = [], [], [], []

        for x_img in val_loader:
            with torch.no_grad():
                x_img = x_img.to(device)
                fx = model(x_img)

            fx_np_raw = fx.squeeze().cpu().numpy()
            x_np_raw = x_img.squeeze().cpu().numpy()

            x_np = crop_center(x_np_raw)
            fx_np = crop_center(fx_np_raw)
            fx_np = np.clip(fx_np, 0, 1)
            x_np = np.clip(x_np, 0, 1)

            val_losses.append(loss_fn(fx, x_img).item())
            val_mses.append(np.mean((fx_np - x_np) ** 2))
            val_ssims.append(ssim_metric(x_np, fx_np, data_range=1.0))
            val_psnrs.append(psnr_metric(x_np, fx_np, data_range=1.0))

        x_img_tensor = torch.tensor(x_np_raw).unsqueeze(0).unsqueeze(0).to(device)
        fx_tensor = torch.tensor(fx_np_raw).unsqueeze(0).unsqueeze(0).to(device)

        h_x = crop_center(apply_psf_torch(x_img_tensor, psf_tensor).squeeze().cpu().numpy())
        h_fx = crop_center(apply_psf_torch(fx_tensor, psf_tensor).squeeze().cpu().numpy())
        diff = np.clip(np.abs(x_np - h_fx), 0, 1)

        ssim_hx = ssim_metric(x_np, h_x, data_range=1.0)
        mse_hx = np.mean((x_np - h_x) ** 2)
        psnr_hx = psnr_metric(x_np, h_x, data_range=1.0)

        ssim_hfx = ssim_metric(x_np, h_fx, data_range=1.0)
        mse_hfx = np.mean((x_np - h_fx) ** 2)
        psnr_hfx = psnr_metric(x_np, h_fx, data_range=1.0)

        print(f"[Amp {amplitud:.2f}] Epoch {epoch} | "
        f"Train MSE: {np.mean(train_mses):.6f} | Val SSIM: {np.mean(val_ssims):.4f} | "
        f"PSNR h*x: {psnr_hx:.4f} | PSNR h*f(x): {psnr_hfx:.4f}")


        wandb.log({
            "amplitud": amplitud,
            "epoch": epoch,
            "train_loss": total_loss / len(train_loader),
            "val_loss": np.mean(val_losses),
            "train_mse": np.mean(train_mses),
            "train_ssim": np.mean(train_ssims),
            "train_psnr": np.mean(train_psnrs),
            "val_mse": np.mean(val_mses),
            "val_ssim": np.mean(val_ssims),
            "val_psnr": np.mean(val_psnrs),
            "hx_ssim": ssim_hx,
            "hx_mse": mse_hx,
            "hx_psnr": psnr_hx,
            "hfx_ssim": ssim_hfx,
            "hfx_mse": mse_hfx,
            "hfx_psnr": psnr_hfx
        })

        #print(f"[Amp {amplitud:.2f}] Epoch {epoch} | Train MSE: {np.mean(train_mses):.6f} | Val SSIM: {np.mean(val_ssims):.4f}")

        fig, axs = plt.subplots(1, 5, figsize=(18, 4))
        axs[0].imshow(x_np, cmap='gray')
        axs[0].set_title("x (original)")
        axs[1].imshow(h_x, cmap='gray')
        axs[1].set_title(f"h*x\nSSIM: {ssim_hx:.2f}\nMSE: {mse_hx:.4f}\nPSNR:FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF {psnr_hx:.1f}")
        axs[2].imshow(fx_np, cmap='gray')
        axs[2].set_title("f(x)")
        axs[3].imshow(h_fx, cmap='gray')
        axs[3].set_title(f"h*f(x)\nSSIM: {ssim_hfx:.2f}\nMSE: {mse_hfx:.4f}\nPSNR: {psnr_hfx:.1f}")
        axs[4].imshow(diff, cmap='gray')
        axs[4].set_title("|x - h*f(x)|")
        for ax in axs:F
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"epoca_{epoch:03d}.png"))
        plt.close()

    torch.save(model.state_dict(), os.path.join(output_dir, "modelo_final.pt"))
    wandb.finish()
