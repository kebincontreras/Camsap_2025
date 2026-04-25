import os
import warnings
warnings.filterwarnings("ignore")
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, Grayscale, Resize
from torch.utils.data import DataLoader, Dataset
from skimage.metrics import structural_similarity as ssim_metric, peak_signal_noise_ratio as psnr_metric
import lpips
import torch.nn.functional as F

from Resources.Ultris.Ultris_model import UNet
from Resources.Ultris.Ultris_zernike import generate_zernike_map, generate_psf, device
from tqdm import tqdm

# --- Configuraciones ---
n, m = 2, 0  # Aberración: Miopía
image_dir = "Resources/Images_kity"
weights_dir = "Resources/weights"
output_root = "Resources/evaluaciones"
os.makedirs(output_root, exist_ok=True)

amplitudes = [0.5, 1.0, 2.0, 3.0]
resize_to_512 = Resize((512, 512))
CROP = 20

# Inicializar modelo LPIPS
print("Cargando modelo LPIPS (AlexNet)...")
lpips_fn = lpips.LPIPS(net='alex').to(device)
lpips_fn.eval()

# --- Funciones auxiliares ---
def crop_center(img):
    return img[CROP:-CROP, CROP:-CROP]

def apply_psf_torch(image_tensor, psf_tensor):
    if psf_tensor.max() == 1.0 and psf_tensor.sum() == 1.0 and torch.count_nonzero(psf_tensor) == 1:
        return image_tensor
    return F.conv2d(image_tensor, psf_tensor, padding="same")

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

# --- Dataset de validación (usando las mismas 3 imágenes que el entrenamiento) ---
all_image_paths = [os.path.join(image_dir, f"{i:06d}.png") for i in range(10)]
val_paths = all_image_paths[7:]

if not val_paths:
    raise ValueError(f"No se encontraron imágenes de validación en {image_dir}")

val_loader = DataLoader(ImageDataset(val_paths), batch_size=1, shuffle=False)

# Archivo de resumen global
summary_txt_path = os.path.join(output_root, "resumen_global.txt")

with open(summary_txt_path, "w", encoding="utf-8") as f_summary:
    f_summary.write("=== Resumen Global de Evaluaciones ===\n\n")

    print("\n[+] Iniciando proceso de evaluación para todas las amplitudes...")
    # Iterar sobre las diferentes severidades (amplitudes)
    for amplitud in amplitudes:
        
        weight_path = os.path.join(weights_dir, f"modelo_final_{amplitud}.pt")
        # Nota: ajustamos nombre_archivo en caso de ser "1" en lugar de "1.0"
        if amplitud == 1.0 and not os.path.exists(weight_path):
            weight_path = os.path.join(weights_dir, "modelo_final_1.pt")
        elif amplitud == 2.0 and not os.path.exists(weight_path):
            weight_path = os.path.join(weights_dir, "modelo_final_2.pt")
        elif amplitud == 3.0 and not os.path.exists(weight_path):
            weight_path = os.path.join(weights_dir, "modelo_final_3.pt")

        if not os.path.exists(weight_path):
            print(f"  [!] No se encontraron pesos en {weight_path}. Saltando...")
            continue

        output_dir = os.path.join(output_root, f"amplitud_{amplitud}")
        os.makedirs(output_dir, exist_ok=True)

        # 1. Cargar el modelo UNet
        model = UNet(in_channels=1, out_channels=1).to(device)
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.eval()
        # print(f"  -> Modelo cargado desde: {weight_path}")

        # 2. Generar el mapa Zernike y la PSF para la aberración actual
        zmap = generate_zernike_map(n, m, amplitude=amplitud)
        psf = generate_psf(zmap)
        psf_tensor = psf.unsqueeze(0).unsqueeze(0).to(device)

        val_ssims_hx, val_psnrs_hx, val_lpips_hx = [], [], []
        val_ssims_wiener, val_psnrs_wiener, val_lpips_wiener = [], [], []
        val_ssims_hfx, val_psnrs_hfx, val_lpips_hfx = [], [], []

        from scipy.signal import wiener

        # 3. Evaluar cada imagen de validación
        for idx, x_img in enumerate(tqdm(val_loader, desc=f"Evaluando Amplitud {amplitud}")):
            with torch.no_grad():
                x_img = x_img.to(device)
                fx = model(x_img) # Imagen transformada f(x)

            x_np_raw = x_img.squeeze().cpu().numpy()
            fx_np_raw = fx.squeeze().cpu().numpy()

            # --- Corrección Analítica (De-convolución Wiener) ---
            from skimage.restoration import wiener
            
            psf_np = psf_tensor.squeeze().cpu().numpy()
            # El parámetro balance representa el NSR (ruido esperado), 0.01 es estándar
            fx_wiener_np_raw = wiener(x_np_raw, psf_np, balance=0.01)
            fx_wiener_np_raw = np.clip(fx_wiener_np_raw, 0, 1).astype(np.float32)

            x_np = crop_center(x_np_raw)
            fx_np = crop_center(fx_np_raw)
            fx_wiener_np = crop_center(fx_wiener_np_raw)

            fx_np = np.clip(fx_np, 0, 1)
            x_np = np.clip(x_np, 0, 1)

            # Aplicar la aberración (PSF) para obtener h*x y h*f(x)
            x_img_tensor = torch.tensor(x_np_raw).unsqueeze(0).unsqueeze(0).to(device)
            fx_tensor = torch.tensor(fx_np_raw).unsqueeze(0).unsqueeze(0).to(device)
            fx_wiener_tensor = torch.tensor(fx_wiener_np_raw).unsqueeze(0).unsqueeze(0).to(device)

            h_x = crop_center(apply_psf_torch(x_img_tensor, psf_tensor).squeeze().cpu().numpy())
            h_fx = crop_center(apply_psf_torch(fx_tensor, psf_tensor).squeeze().cpu().numpy())
            h_fx_wiener = crop_center(apply_psf_torch(fx_wiener_tensor, psf_tensor).squeeze().cpu().numpy())
            
            h_x = np.clip(h_x, 0, 1)
            h_fx = np.clip(h_fx, 0, 1)
            h_fx_wiener = np.clip(h_fx_wiener, 0, 1)
            diff_abs = np.clip(np.abs(x_np - h_fx), 0, 1)
            diff_abs_wiener = np.clip(np.abs(x_np - h_fx_wiener), 0, 1)

            # Calcular métricas base para h*x vs x (Baseline)
            ssim_hx = ssim_metric(x_np, h_x, data_range=1.0)
            psnr_hx = psnr_metric(x_np, h_x, data_range=1.0)

            # Calcular métricas de la compensación para h*f(x) vs x (Resultado)
            ssim_hfx = ssim_metric(x_np, h_fx, data_range=1.0)
            psnr_hfx = psnr_metric(x_np, h_fx, data_range=1.0)

            # Calcular métricas para Wiener
            ssim_wiener = ssim_metric(x_np, h_fx_wiener, data_range=1.0)
            psnr_wiener = psnr_metric(x_np, h_fx_wiener, data_range=1.0)

            # --- Cálculo de LPIPS ---
            x_lpips = torch.tensor(x_np).unsqueeze(0).unsqueeze(0).to(device)
            x_lpips = x_lpips * 2.0 - 1.0  # Rango [-1, 1]
            x_lpips = x_lpips.repeat(1, 3, 1, 1)  # 1ch -> 3ch

            hx_lpips = torch.tensor(h_x).unsqueeze(0).unsqueeze(0).to(device)
            hx_lpips = hx_lpips * 2.0 - 1.0
            hx_lpips = hx_lpips.repeat(1, 3, 1, 1)
            
            hfx_lpips = torch.tensor(h_fx).unsqueeze(0).unsqueeze(0).to(device)
            hfx_lpips = hfx_lpips * 2.0 - 1.0
            hfx_lpips = hfx_lpips.repeat(1, 3, 1, 1)
            
            hwiener_lpips = torch.tensor(h_fx_wiener).unsqueeze(0).unsqueeze(0).to(device)
            hwiener_lpips = hwiener_lpips * 2.0 - 1.0
            hwiener_lpips = hwiener_lpips.repeat(1, 3, 1, 1)

            with torch.no_grad():
                lpips_hx = lpips_fn(x_lpips, hx_lpips).item()
                lpips_hfx = lpips_fn(x_lpips, hfx_lpips).item()
                lpips_wiener = lpips_fn(x_lpips, hwiener_lpips).item()

            val_ssims_hx.append(ssim_hx)
            val_psnrs_hx.append(psnr_hx)
            val_lpips_hx.append(lpips_hx)

            val_ssims_hfx.append(ssim_hfx)
            val_psnrs_hfx.append(psnr_hfx)
            val_lpips_hfx.append(lpips_hfx)
            
            val_ssims_wiener.append(ssim_wiener)
            val_psnrs_wiener.append(psnr_wiener)
            val_lpips_wiener.append(lpips_wiener)

            # Guardar figura comparativa de la imagen actual
            fig, axs = plt.subplots(1, 6, figsize=(24, 5))
            axs[0].imshow(x_np, cmap='gray')
            axs[0].set_title("Original (x)")
            
            axs[1].imshow(h_x, cmap='gray')
            axs[1].set_title(f"Baseline (Vista Miope)\nPSNR: {psnr_hx:.1f} | SSIM: {ssim_hx:.2f}\nLPIPS: {lpips_hx:.3f}")
            
            axs[2].imshow(h_fx_wiener, cmap='gray')
            axs[2].set_title(f"Filtro Wiener (h*f_wiener)\nPSNR: {psnr_wiener:.1f} | SSIM: {ssim_wiener:.2f}\nLPIPS: {lpips_wiener:.3f}")

            axs[3].imshow(fx_np, cmap='gray')
            axs[3].set_title("Pre-compensada UNet (f(x))")
            
            axs[4].imshow(h_fx, cmap='gray')
            axs[4].set_title(f"UNet Corregida (h*f(x))\nPSNR: {psnr_hfx:.1f} | SSIM: {ssim_hfx:.2f}\nLPIPS: {lpips_hfx:.3f}")
            
            axs[5].imshow(diff_abs, cmap='gray')
            axs[5].set_title("|x - h*f(x)| (Diff UNet)")
            
            for ax in axs:
                ax.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"eval_imagen_{idx+1}.png"), dpi=150)
            plt.close()

        # Guardar resumen por amplitud
        def get_stats(arr):
            arr = np.array(arr)
            return np.mean(arr), np.std(arr), np.min(arr), np.max(arr)

        mean_psnr_hx, std_psnr_hx, min_psnr_hx, max_psnr_hx = get_stats(val_psnrs_hx)
        mean_ssim_hx, std_ssim_hx, min_ssim_hx, max_ssim_hx = get_stats(val_ssims_hx)
        mean_lpips_hx, std_lpips_hx, min_lpips_hx, max_lpips_hx = get_stats(val_lpips_hx)

        mean_psnr_hfx, std_psnr_hfx, min_psnr_hfx, max_psnr_hfx = get_stats(val_psnrs_hfx)
        mean_ssim_hfx, std_ssim_hfx, min_ssim_hfx, max_ssim_hfx = get_stats(val_ssims_hfx)
        mean_lpips_hfx, std_lpips_hfx, min_lpips_hfx, max_lpips_hfx = get_stats(val_lpips_hfx)

        mean_psnr_wiener, std_psnr_wiener, min_psnr_wiener, max_psnr_wiener = get_stats(val_psnrs_wiener)
        mean_ssim_wiener, std_ssim_wiener, min_ssim_wiener, max_ssim_wiener = get_stats(val_ssims_wiener)
        mean_lpips_wiener, std_lpips_wiener, min_lpips_wiener, max_lpips_wiener = get_stats(val_lpips_wiener)

        report = f"""Resultados Amplitud {amplitud}:

| Métrica | Condición | Promedio | Desv. Est. | Mínimo | Máximo |
|---------|-----------|----------|------------|--------|--------|
| PSNR    | Baseline  | {mean_psnr_hx:>8.2f} | {std_psnr_hx:>10.2f} | {min_psnr_hx:>6.2f} | {max_psnr_hx:>6.2f} |
| PSNR    | Wiener    | {mean_psnr_wiener:>8.2f} | {std_psnr_wiener:>10.2f} | {min_psnr_wiener:>6.2f} | {max_psnr_wiener:>6.2f} |
| PSNR    | UNet      | {mean_psnr_hfx:>8.2f} | {std_psnr_hfx:>10.2f} | {min_psnr_hfx:>6.2f} | {max_psnr_hfx:>6.2f} |
| SSIM    | Baseline  | {mean_ssim_hx:>8.4f} | {std_ssim_hx:>10.4f} | {min_ssim_hx:>6.4f} | {max_ssim_hx:>6.4f} |
| SSIM    | Wiener    | {mean_ssim_wiener:>8.4f} | {std_ssim_wiener:>10.4f} | {min_ssim_wiener:>6.4f} | {max_ssim_wiener:>6.4f} |
| SSIM    | UNet      | {mean_ssim_hfx:>8.4f} | {std_ssim_hfx:>10.4f} | {min_ssim_hfx:>6.4f} | {max_ssim_hfx:>6.4f} |
| LPIPS   | Baseline  | {mean_lpips_hx:>8.4f} | {std_lpips_hx:>10.4f} | {min_lpips_hx:>6.4f} | {max_lpips_hx:>6.4f} |
| LPIPS   | Wiener    | {mean_lpips_wiener:>8.4f} | {std_lpips_wiener:>10.4f} | {min_lpips_wiener:>6.4f} | {max_lpips_wiener:>6.4f} |
| LPIPS   | UNet      | {mean_lpips_hfx:>8.4f} | {std_lpips_hfx:>10.4f} | {min_lpips_hfx:>6.4f} | {max_lpips_hfx:>6.4f} |

Mejora UNet vs Baseline (Δ):
  Δ PSNR : {mean_psnr_hfx - mean_psnr_hx:+.2f}
  Δ SSIM : {mean_ssim_hfx - mean_ssim_hx:+.4f}
  Δ LPIPS: {mean_lpips_hfx - mean_lpips_hx:+.4f} (Negativo es mejor)
  
Mejora UNet vs Wiener (Δ):
  Δ PSNR : {mean_psnr_hfx - mean_psnr_wiener:+.2f}
  Δ SSIM : {mean_ssim_hfx - mean_ssim_wiener:+.4f}
  Δ LPIPS: {mean_lpips_hfx - mean_lpips_wiener:+.4f} (Negativo es mejor)
{"-"*70}
"""
        
        # Escribir en txt individual
        with open(os.path.join(output_dir, "metricas.txt"), "w", encoding="utf-8") as f:
            f.write(report)
        
        # Escribir en txt global
        f_summary.write(report + "\n")

print(f"\n[+] Evaluación completada. Revisa la carpeta '{output_root}'.")
