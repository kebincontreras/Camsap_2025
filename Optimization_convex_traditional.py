import os
import torch
import numpy as np
import wandb
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, Grayscale, Resize
from torch.utils.data import DataLoader, Dataset
from skimage.metrics import structural_similarity as ssim_metric, peak_signal_noise_ratio as psnr_metric
from scipy.signal import wiener
from Resources.Ultris.Ultris_zernike import generate_zernike_map, generate_psf, apply_psf, device

# Configuraciones generales
n, m = 2, 2
image_dir = "Images"
output_root = "resultados_wiener"
os.makedirs(output_root, exist_ok=True)

resize_to_512 = Resize((256, 256))  # Ajusta tamaño según necesites

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
val_paths = all_image_paths[7:]  # Solo validación

val_loader = DataLoader(ImageDataset(val_paths), batch_size=1, shuffle=False)

# Recorte centrado
CROP = 20
def crop_center(img):
    return img[CROP:-CROP, CROP:-CROP]

# Función para aplicar PSF en tensor
def apply_psf_torch(image_tensor, psf_tensor):
    if psf_tensor.max() == 1.0 and psf_tensor.sum() == 1.0 and torch.count_nonzero(psf_tensor) == 1:
        return image_tensor
    return torch.nn.functional.conv2d(image_tensor, psf_tensor, padding="same")

# Lista de severidades a procesar
amplitudes = [0.5, 1.0, 2.0, 3.0]

for amplitud in amplitudes:
    print(f"Procesando para amplitud {amplitud}")
    wandb.init(project="restauracion-zernike", config={"n": n, "m": m, "amplitud": amplitud}, dir="Resources")

    # Crear PSF y tensor
    zmap = generate_zernike_map(n, m, amplitude=amplitud)
    psf = generate_psf(zmap).cpu().numpy()
    psf_tensor = torch.tensor(psf).unsqueeze(0).unsqueeze(0).to(device)

    output_dir = os.path.join(output_root, f"amplitud_{amplitud:.2f}")
    os.makedirs(output_dir, exist_ok=True)

    val_mses, val_ssims, val_psnrs = [], [], []

    for idx, x_img in enumerate(val_loader):
        x_img = x_img.to(device)
        x_np_raw = x_img.squeeze().cpu().numpy()
        x_np = crop_center(x_np_raw)
        x_np = np.clip(x_np, 0, 1)

        # Aplicar PSF (imagen borrosa)
        x_img_tensor = torch.tensor(x_np_raw).unsqueeze(0).unsqueeze(0).to(device)
        h_x = crop_center(apply_psf_torch(x_img_tensor, psf_tensor).squeeze().cpu().numpy())

        # Aplicar filtro Wiener sobre la imagen borrosa
        restored = wiener(h_x, mysize=(5,5), noise=None)
        restored = np.clip(restored, 0, 1)

        # Calcular métricas
        mse = np.mean((restored - x_np) ** 2)
        ssim_val = ssim_metric(x_np, restored, data_range=1.0)

        # PSNR h*x (borrosa vs original)
        psnr_hx = psnr_metric(x_np, h_x, data_range=1.0)
        # PSNR f(x) (restaurada vs original)
        psnr_restored = psnr_metric(x_np, restored, data_range=1.0)

        val_mses.append(mse)
        val_ssims.append(ssim_val)
        val_psnrs.append(psnr_restored)

        print(f"[Amp {amplitud:.2f}] Img {idx+1} | PSNR h*x: {psnr_hx:.4f} | PSNR f(x): {psnr_restored:.4f} | MSE: {mse:.6f} | SSIM: {ssim_val:.4f}")

        # Diferencia para visualización
        diff = np.clip(np.abs(x_np - restored), 0, 1)

        # Guardar figura con imágenes y métricas
        fig, axs = plt.subplots(1, 4, figsize=(15, 4))
        axs[0].imshow(x_np, cmap='gray')
        axs[0].set_title("x (original)")
        axs[1].imshow(h_x, cmap='gray')
        axs[1].set_title("h*x (borrosa)")
        axs[2].imshow(restored, cmap='gray')
        axs[2].set_title("Restaurada (Wiener)")
        axs[3].imshow(diff, cmap='gray')
        axs[3].set_title("|x - restaurada|")

        for ax in axs:
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"imagen_{idx+1}.png"))
        plt.close()

    wandb.log({
        "amplitud": amplitud,
        "val_mse": np.mean(val_mses),
        "val_ssim": np.mean(val_ssims),
        "val_psnr": np.mean(val_psnrs)
    })

    wandb.finish()
