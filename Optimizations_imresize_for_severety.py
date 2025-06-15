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
from Resources.Ultris.Ultris_model import SimpleCNN
from Resources.Ultris.Ultris_zernike import generate_zernike_map, generate_psf, apply_psf, device
import torch.nn.functional as F
import cv2
from skimage.restoration import richardson_lucy
from losses import loss_functions

# Configuraciones generales
n, m = 2, 0
num_epochs = 2
image_dir = "Resources/Images_kity"
output_root = "Resources/resultados_epocas"
os.makedirs(output_root, exist_ok=True)

resize_to_512 = Resize((512, 512))

# Función para aplicar filtros
def apply_filter(x, filter_name, psf_tensor=None):
    x_np = x.squeeze().cpu().numpy()
    if filter_name == 'wiener':
        from scipy.signal import wiener
        x_filt = wiener(x_np, (5, 5))
    elif filter_name == 'median':
        from scipy.ndimage import median_filter
        x_filt = median_filter(x_np, size=3)
    elif filter_name == 'gaussian':
        from scipy.ndimage import gaussian_filter
        x_filt = gaussian_filter(x_np, sigma=1)
    elif filter_name == 'bilateral':
        x_uint8 = (x_np * 255).astype(np.uint8)
        x_filt = cv2.bilateralFilter(x_uint8, d=5, sigmaColor=75, sigmaSpace=75)
        x_filt = x_filt.astype(np.float32) / 255.0
    elif filter_name == 'richardson_lucy':
        psf_np = psf_tensor.squeeze().cpu().numpy()
        x_filt = richardson_lucy(x_np, psf_np, num_iter=10)
    elif filter_name == 'none':
        x_filt = x_np
    else:
        raise ValueError(f"Filtro no soportado: {filter_name}")
    x_filt = torch.tensor(x_filt, dtype=x.dtype).unsqueeze(0).unsqueeze(0).to(x.device)
    return x_filt

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
amplitudes = [0.5]

# Lista de nombres de experimentos (losses) a probar
experiment_names = list(loss_functions.keys())

# ...existing code...

for amplitud in amplitudes:
    print(f"Entrenando para amplitud {amplitud}")
    wandb.init(project="restauracion-zernike", config={"n": n, "m": m, "amplitud": amplitud, "epochs": num_epochs}, dir="Resources")

    # Crear PSF
    zmap = generate_zernike_map(n, m, amplitude=amplitud)
    psf = generate_psf(zmap)
    psf_tensor = psf.unsqueeze(0).unsqueeze(0).to(device)
    h = lambda x: apply_psf_torch(x, psf_tensor)

    # Archivo de resultados para esta amplitud
    results_txt = os.path.join(output_root, f"resultados_amplitud_{amplitud:.2f}.txt")
    with open(results_txt, "a") as ftxt:

        for experiment_name in experiment_names:
            print(f"  -> Experimento: {experiment_name}")
            ftxt.write(f"\n=== Experimento: {experiment_name} ===\n")
            output_dir = os.path.join(output_root, f"amplitud_{amplitud:.2f}", experiment_name)
            os.makedirs(output_dir, exist_ok=True)

            # ...inicialización de modelos y optimizador igual...

            if "cnn0_unet" in experiment_name or "unet_cnn0" in experiment_name:
                unet = UNet(in_channels=1, out_channels=1).to(device)
                cnn0 = SimpleCNN(in_channels=1, out_channels=1).to(device)
                if experiment_name.startswith("cnn0_unet"):
                    model_args = (unet, cnn0)
                else:
                    model_args = (cnn0, unet)
                optimizer = torch.optim.Adam(list(unet.parameters()) + list(cnn0.parameters()), lr=1e-3)
                def combined_forward(x):
                    return unet(x) + cnn0(x)
                main_model = combined_forward
            elif "cnn0" in experiment_name:
                cnn0 = SimpleCNN(in_channels=1, out_channels=1).to(device)
                model_args = (cnn0,)
                optimizer = torch.optim.Adam(cnn0.parameters(), lr=1e-3)
                main_model = cnn0
            else:
                unet = UNet(in_channels=1, out_channels=1).to(device)
                model_args = (unet,)
                optimizer = torch.optim.Adam(unet.parameters(), lr=1e-3)
                main_model = unet

            loss_fn = loss_functions[experiment_name]

            for epoch in range(1, num_epochs + 1):
                if "cnn0_unet" in experiment_name or "unet_cnn0" in experiment_name:
                    unet.train()
                    cnn0.train()
                else:
                    main_model.train()
                total_loss = 0.0
                train_mses, train_ssims, train_psnrs = [], [], []

                for x in train_loader:
                    x = x.to(device)
                    loss = loss_fn(*model_args, h, x, apply_filter)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                    with torch.no_grad():
                        if "cnn0_unet" in experiment_name or "unet_cnn0" in experiment_name:
                            fx = main_model(x)
                        else:
                            fx = main_model(x)
                    fx_np = crop_center(fx.squeeze().detach().cpu().numpy())
                    x_np = crop_center(x.squeeze().detach().cpu().numpy())
                    fx_np = np.clip(fx_np, 0, 1)
                    x_np = np.clip(x_np, 0, 1)
                    train_mses.append(np.mean((fx_np - x_np)**2))
                    train_ssims.append(ssim_metric(x_np, fx_np, data_range=1.0))
                    train_psnrs.append(psnr_metric(x_np, fx_np, data_range=1.0))

                # Evaluación
                if "cnn0_unet" in experiment_name or "unet_cnn0" in experiment_name:
                    unet.eval()
                    cnn0.eval()
                else:
                    main_model.eval()
                val_losses, val_mses, val_ssims, val_psnrs = [], [], [], []

                for x_img in val_loader:
                    with torch.no_grad():
                        x_img = x_img.to(device)
                        if "cnn0_unet" in experiment_name or "unet_cnn0" in experiment_name:
                            fx = main_model(x_img)
                        else:
                            fx = main_model(x_img)

                    fx_np_raw = fx.squeeze().cpu().numpy()
                    x_np_raw = x_img.squeeze().cpu().numpy()

                    x_np = crop_center(x_np_raw)
                    fx_np = crop_center(fx_np_raw)
                    fx_np = np.clip(fx_np, 0, 1)
                    x_np = np.clip(x_np, 0, 1)

                    val_losses.append(F.mse_loss(fx, x_img).item())
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

                log_str = (f"[Amp {amplitud:.2f}][{experiment_name}] Epoch {epoch} | "
                           f"Train MSE: {np.mean(train_mses):.6f} | Val SSIM: {np.mean(val_ssims):.4f} | "
                           f"PSNR h*x: {psnr_hx:.4f} | PSNR h*f(x): {psnr_hfx:.4f}")
                print(log_str)
                ftxt.write(log_str + "\n")
                ftxt.flush()

                wandb.log({
                    "amplitud": amplitud,
                    "experiment": experiment_name,
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

                fig, axs = plt.subplots(1, 5, figsize=(18, 4))
                axs[0].imshow(x_np, cmap='gray')
                axs[0].set_title("x (original)")
                axs[1].imshow(h_x, cmap='gray')
                axs[1].set_title(f"h*x\nSSIM: {ssim_hx:.2f}\nMSE: {mse_hx:.4f}\nPSNR: {psnr_hx:.1f}")
                axs[2].imshow(fx_np, cmap='gray')
                axs[2].set_title("f(x)")
                axs[3].imshow(h_fx, cmap='gray')
                axs[3].set_title(f"h*f(x)\nSSIM: {ssim_hfx:.2f}\nMSE: {mse_hfx:.4f}\nPSNR: {psnr_hfx:.1f}")
                axs[4].imshow(diff, cmap='gray')
                axs[4].set_title("|x - h*f(x)|")
                for ax in axs:
                    ax.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"epoca_{epoch:03d}.png"))
                plt.close()

            # Guardar ambos modelos si aplica
            if "cnn0_unet" in experiment_name or "unet_cnn0" in experiment_name:
                torch.save(unet.state_dict(), os.path.join(output_dir, "unet_final.pt"))
                torch.save(cnn0.state_dict(), os.path.join(output_dir, "cnn0_final.pt"))
            else:
                torch.save(main_model.state_dict(), os.path.join(output_dir, "modelo_final.pt"))
    wandb.finish()