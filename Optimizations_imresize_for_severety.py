import os
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("wandb").setLevel(logging.ERROR)
import torch
import numpy as np
import wandb
from PIL import Image
from torchvision.transforms import ToTensor, Grayscale, Resize
from torch.utils.data import DataLoader, Dataset
from skimage.metrics import structural_similarity as ssim_metric, peak_signal_noise_ratio as psnr_metric
import lpips
from Resources.Ultris.Ultris_model import UNet
from Resources.Ultris.Ultris_model import SimpleCNN
from Resources.Ultris.Ultris_zernike import generate_zernike_map, generate_psf, apply_psf, device
import torch.nn.functional as F
import cv2
from skimage.restoration import richardson_lucy
from tqdm import tqdm
from losses import loss_functions

# Configuraciones generales
n, m = 2, 0
num_epochs = 50
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

# Listar todas las imagenes de KITTI disponibles
import random
kitti_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])
all_image_paths = [os.path.join(image_dir, f) for f in kitti_files]

# Excluir los 3 fijos de validacion (000007, 000008, 000009)
val_fijos = ['000007.png', '000008.png', '000009.png']
candidatos_train = [p for p in all_image_paths if os.path.basename(p) not in val_fijos]

# Tomar aleatoriamente 80 para train y 20 para val del resto
random.seed(42)
random.shuffle(candidatos_train)
train_paths = candidatos_train[:80]
val_paths   = candidatos_train[80:100]  # 20 imagenes de validacion

print(f"Dataset: {len(train_paths)} train | {len(val_paths)} val")

train_loader = DataLoader(ImageDataset(train_paths), batch_size=1, shuffle=True)
val_loader   = DataLoader(ImageDataset(val_paths),   batch_size=1, shuffle=False)


# Recorte centrado
CROP = 20
def crop_center(img):
    return img[CROP:-CROP, CROP:-CROP]

# Función para aplicar PSF
def apply_psf_torch(image_tensor, psf_tensor):
    if psf_tensor.max() == 1.0 and psf_tensor.sum() == 1.0 and torch.count_nonzero(psf_tensor) == 1:
        return image_tensor
    return F.conv2d(image_tensor, psf_tensor, padding="same")

# Inicializar modelo LPIPS (red AlexNet, más rápida)
lpips_fn = lpips.LPIPS(net='alex').to(device)

# Lista de severidades a procesar
amplitudes = [0.5, 1.0, 2.0, 3.0]

# Lista de nombres de experimentos (losses) a probar
experiment_names = list(loss_functions.keys())

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

                pbar = tqdm(train_loader, desc=f"  Ep {epoch:02d}/{num_epochs}", leave=False, ncols=80)
                for x in pbar:
                    x = x.to(device)
                    loss = loss_fn(*model_args, h, x, apply_filter)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    pbar.set_postfix(loss=f"{loss.item():.5f}")

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

                # LPIPS para h(f(x)) vs x — convierte a tensores [1,3,H,W] en rango [-1,1]
                x_lpips = torch.tensor(x_np).unsqueeze(0).unsqueeze(0).to(device)
                x_lpips = x_lpips * 2.0 - 1.0  # [0,1] -> [-1,1]
                x_lpips = x_lpips.repeat(1, 3, 1, 1)  # 1ch -> 3ch
                hfx_lpips = torch.tensor(h_fx).unsqueeze(0).unsqueeze(0).to(device)
                hfx_lpips = hfx_lpips * 2.0 - 1.0
                hfx_lpips = hfx_lpips.repeat(1, 3, 1, 1)
                with torch.no_grad():
                    lpips_hfx = lpips_fn(x_lpips, hfx_lpips).item()

                log_str = (f"[Amp {amplitud:.2f}][{experiment_name}] Epoch {epoch} | "
                           f"Train MSE: {np.mean(train_mses):.6f} | Val SSIM: {np.mean(val_ssims):.4f} | "
                           f"PSNR h*x: {psnr_hx:.4f} | PSNR h*f(x): {psnr_hfx:.4f} | LPIPS h*f(x): {lpips_hfx:.4f}")
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
                    "hfx_psnr": psnr_hfx,
                    "hfx_lpips": lpips_hfx
                })

            # Guardar ambos modelos si aplica
            if "cnn0_unet" in experiment_name or "unet_cnn0" in experiment_name:
                torch.save(unet.state_dict(), os.path.join(output_dir, "unet_final.pt"))
                torch.save(cnn0.state_dict(), os.path.join(output_dir, "cnn0_final.pt"))
            else:
                torch.save(main_model.state_dict(), os.path.join(output_dir, "modelo_final.pt"))

                # Formato estándar de amplitud: 0.5 queda como "0.5", 1.0 → "1", 2.0 → "2", 3.0 → "3"
                fmt_amp = int(amplitud) if isinstance(amplitud, float) and amplitud.is_integer() else amplitud

                # Guardar clon del peso "unet_x" o "cnn0_x" (Solo-MSE) en la carpeta separada
                if experiment_name == "unet_x":
                    weights_dest = "Resources/weights_MSE_UNET"
                    os.makedirs(weights_dest, exist_ok=True)
                    torch.save(main_model.state_dict(), os.path.join(weights_dest, f"modelo_final_{fmt_amp}.pt"))
                    print(f"  [✓] Pesos UNet MSE guardados en: {weights_dest}/modelo_final_{fmt_amp}.pt")
                elif experiment_name == "cnn0_x":
                    weights_dest = "Resources/weights_mse_cnn"
                    os.makedirs(weights_dest, exist_ok=True)
                    torch.save(main_model.state_dict(), os.path.join(weights_dest, f"modelo_final_{fmt_amp}.pt"))
                    print(f"  [✓] Pesos CNN MSE guardados en: {weights_dest}/modelo_final_{fmt_amp}.pt")

                # Guardar clon del peso "unet_x_high_freq" o "cnn0_x_high_freq" (Loss propuesta: MSE + Sobel) en la carpeta separada
                if experiment_name == "unet_x_high_freq":
                    weights_dest = "Resources/weights_prop_UNET"
                    os.makedirs(weights_dest, exist_ok=True)
                    torch.save(main_model.state_dict(), os.path.join(weights_dest, f"modelo_final_{fmt_amp}.pt"))
                    print(f"  [✓] Pesos UNet Propuesta guardados en: {weights_dest}/modelo_final_{fmt_amp}.pt")
                elif experiment_name == "cnn0_x_high_freq":
                    weights_dest = "Resources/weights_prop_cnn"
                    os.makedirs(weights_dest, exist_ok=True)
                    torch.save(main_model.state_dict(), os.path.join(weights_dest, f"modelo_final_{fmt_amp}.pt"))
                    print(f"  [✓] Pesos CNN Propuesta guardados en: {weights_dest}/modelo_final_{fmt_amp}.pt")
    wandb.finish()