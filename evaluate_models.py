import os
import warnings
warnings.filterwarnings("ignore")
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor, Grayscale, Resize
from torch.utils.data import DataLoader, Dataset
from skimage.metrics import structural_similarity as ssim_metric, peak_signal_noise_ratio as psnr_metric
from skimage.restoration import wiener
import lpips
import torch.nn.functional as F

from Resources.Ultris.Ultris_model import UNet
from Resources.Ultris.Ultris_zernike import generate_zernike_map, generate_psf, device
from tqdm import tqdm

# --- Configuraciones ---
n, m = 2, 0  # Aberración: Miopía
weights_dir         = "Resources/weights_prop_UNET"  # UNet entrenado con pérdida propuesta (MSE + Sobel)
weights_mse_dir     = "Resources/weights_MSE_UNET"  # UNet entrenado solo con MSE
output_root         = "Resources/evaluaciones"
os.makedirs(output_root, exist_ok=True)

amplitudes = [0.5, 1.0, 2.0, 3.0]
resize_to_512 = Resize((512, 512))
CROP = 20

# Diccionario de datasets disponibles para evaluar
datasets_dirs = {
    "KITTI":   "Resources/Images_kity",
    "ImageNet": "Resources/ImageNet",
    "DIV2K":   "Resources/DIV2K"
}

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

def get_dataset_paths(dname, ddir):
    if not os.path.exists(ddir):
        return []
    if dname == "KITTI":
        paths = [os.path.join(ddir, f"{i:06d}.png") for i in range(7, 10)]
        return [p for p in paths if os.path.exists(p)]
    else:
        paths = sorted([os.path.join(ddir, f) for f in os.listdir(ddir) if f.endswith(".png")])
        return paths

class ImageDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return load_image_tensor(self.paths[idx])

def get_weight_path(base_dir, amplitud):
    """Busca el archivo de pesos con fallback para nombres sin decimales."""
    path = os.path.join(base_dir, f"modelo_final_{amplitud}.pt")
    if not os.path.exists(path) and amplitud == 1.0:
        path = os.path.join(base_dir, "modelo_final_1.pt")
    elif not os.path.exists(path) and amplitud == 2.0:
        path = os.path.join(base_dir, "modelo_final_2.pt")
    elif not os.path.exists(path) and amplitud == 3.0:
        path = os.path.join(base_dir, "modelo_final_3.pt")
    return path if os.path.exists(path) else None

def load_unet(weight_path):
    """Carga y devuelve un UNet en modo eval."""
    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    return model

def compute_lpips(arr_a, arr_b):
    """Calcula LPIPS entre dos arrays numpy [H,W] float32 en [0,1]."""
    t_a = torch.tensor(arr_a).unsqueeze(0).unsqueeze(0).to(device)
    t_b = torch.tensor(arr_b).unsqueeze(0).unsqueeze(0).to(device)
    t_a = (t_a * 2.0 - 1.0).repeat(1, 3, 1, 1)
    t_b = (t_b * 2.0 - 1.0).repeat(1, 3, 1, 1)
    with torch.no_grad():
        return lpips_fn(t_a, t_b).item()

def get_stats(arr):
    arr = np.array(arr)
    if len(arr) == 0:
        return 0, 0, 0, 0
    return np.mean(arr), np.std(arr), np.min(arr), np.max(arr)

# ---------------------------------------------------------------------------
# Archivo de resumen global
# ---------------------------------------------------------------------------
summary_txt_path = os.path.join(output_root, "resumen_global.txt")

with open(summary_txt_path, "w", encoding="utf-8") as f_summary:
    f_summary.write("=== Resumen Global de Evaluaciones ===\n\n")
    f_summary.write(
        "Métodos comparados:\n"
        "  1. Baseline   : imagen aberrada h*x vs original x\n"
        "  2. Wiener     : filtro Wiener clásico\n"
        "  3. UNet-MSE   : UNet entrenado SOLO con pérdida MSE\n"
        "  4. UNet-Prop  : UNet entrenado con pérdida propuesta (MSE + Sobel)\n\n"
    )

    print("\n[+] Iniciando proceso de evaluación para todas las amplitudes...")

    for amplitud in amplitudes:

        # --- Cargar pesos UNet-Propuesto ---
        wp_prop = get_weight_path(weights_dir, amplitud)
        if wp_prop is None:
            print(f"  [!] No hay pesos propuestos para amplitud {amplitud} en '{weights_dir}'. Saltando...")
            continue
        model_prop = load_unet(wp_prop)
        print(f"\n[Amp {amplitud}] UNet-Propuesto cargado desde: {wp_prop}")

        # --- Cargar pesos UNet-MSE (opcional, puede no existir aún) ---
        wp_mse = get_weight_path(weights_mse_dir, amplitud)
        model_mse = None
        if wp_mse:
            model_mse = load_unet(wp_mse)
            print(f"[Amp {amplitud}] UNet-MSE      cargado desde: {wp_mse}")
        else:
            print(f"[Amp {amplitud}] UNet-MSE no encontrado en '{weights_mse_dir}'. "
                  f"Se omitirá esa columna para esta amplitud.")

        # --- Generar la PSF para esta amplitud ---
        zmap = generate_zernike_map(n, m, amplitude=amplitud)
        psf  = generate_psf(zmap)
        psf_tensor = psf.unsqueeze(0).unsqueeze(0).to(device)

        output_dir = os.path.join(output_root, f"amplitud_{amplitud}")
        os.makedirs(output_dir, exist_ok=True)

        # --- Iterar datasets ---
        for dname, ddir in datasets_dirs.items():
            val_paths = get_dataset_paths(dname, ddir)
            if not val_paths:
                print(f"  [!] Dataset {dname} no encontrado en {ddir}. Saltando...")
                continue

            val_loader = DataLoader(ImageDataset(val_paths), batch_size=1, shuffle=False)

            ds_output_dir = os.path.join(output_dir, dname)
            os.makedirs(ds_output_dir, exist_ok=True)

            # Acumuladores de métricas
            psnrs_hx,      ssims_hx,      lpips_hx_list      = [], [], []
            psnrs_wiener,  ssims_wiener,  lpips_wiener_list   = [], [], []
            psnrs_mse,     ssims_mse,     lpips_mse_list      = [], [], []
            psnrs_prop,    ssims_prop,    lpips_prop_list     = [], [], []

            for idx, x_img in enumerate(tqdm(val_loader, desc=f"Amp {amplitud} | {dname}")):
                x_img = x_img.to(device)

                with torch.no_grad():
                    fx_prop = model_prop(x_img)
                    fx_mse  = model_mse(x_img) if model_mse is not None else None

                x_np_raw    = x_img.squeeze().cpu().numpy()
                fx_prop_raw = fx_prop.squeeze().cpu().numpy()
                fx_mse_raw  = fx_mse.squeeze().cpu().numpy() if fx_mse is not None else None

                # --- Wiener ---
                psf_np        = psf_tensor.squeeze().cpu().numpy()
                fx_wiener_raw = np.clip(wiener(x_np_raw, psf_np, balance=0.01), 0, 1).astype(np.float32)

                # --- Aplicar PSF (h * salida) y recortar bordes ---
                def to_hfx(arr_raw):
                    t = torch.tensor(arr_raw).unsqueeze(0).unsqueeze(0).to(device)
                    return crop_center(apply_psf_torch(t, psf_tensor).squeeze().cpu().numpy())

                h_x      = np.clip(to_hfx(x_np_raw),    0, 1)
                h_wiener = np.clip(to_hfx(fx_wiener_raw), 0, 1)
                h_prop   = np.clip(to_hfx(fx_prop_raw),  0, 1)
                h_mse    = np.clip(to_hfx(fx_mse_raw),   0, 1) if fx_mse_raw is not None else None

                x_np = np.clip(crop_center(x_np_raw), 0, 1)

                # --- PSNR + SSIM ---
                psnrs_hx.append(psnr_metric(x_np, h_x,      data_range=1.0))
                ssims_hx.append(ssim_metric(x_np, h_x,      data_range=1.0))

                psnrs_wiener.append(psnr_metric(x_np, h_wiener, data_range=1.0))
                ssims_wiener.append(ssim_metric(x_np, h_wiener, data_range=1.0))

                psnrs_prop.append(psnr_metric(x_np, h_prop,  data_range=1.0))
                ssims_prop.append(ssim_metric(x_np, h_prop,  data_range=1.0))

                if h_mse is not None:
                    psnrs_mse.append(psnr_metric(x_np, h_mse, data_range=1.0))
                    ssims_mse.append(ssim_metric(x_np, h_mse, data_range=1.0))

                # --- LPIPS ---
                lpips_hx_list.append(compute_lpips(x_np, h_x))
                lpips_wiener_list.append(compute_lpips(x_np, h_wiener))
                lpips_prop_list.append(compute_lpips(x_np, h_prop))
                if h_mse is not None:
                    lpips_mse_list.append(compute_lpips(x_np, h_mse))

            # --- Estadísticas ---
            mp_hx, sp_hx, mnp_hx, mxp_hx           = get_stats(psnrs_hx)
            ms_hx, ss_hx, mns_hx, mxs_hx           = get_stats(ssims_hx)
            ml_hx, sl_hx, mnl_hx, mxl_hx           = get_stats(lpips_hx_list)

            mp_wi, sp_wi, mnp_wi, mxp_wi            = get_stats(psnrs_wiener)
            ms_wi, ss_wi, mns_wi, mxs_wi            = get_stats(ssims_wiener)
            ml_wi, sl_wi, mnl_wi, mxl_wi            = get_stats(lpips_wiener_list)

            mp_pr, sp_pr, mnp_pr, mxp_pr            = get_stats(psnrs_prop)
            ms_pr, ss_pr, mns_pr, mxs_pr            = get_stats(ssims_prop)
            ml_pr, sl_pr, mnl_pr, mxl_pr            = get_stats(lpips_prop_list)

            mp_ms, sp_ms, mnp_ms, mxp_ms            = get_stats(psnrs_mse)
            ms_ms, ss_ms, mns_ms, mxs_ms            = get_stats(ssims_mse)
            ml_ms, sl_ms, mnl_ms, mxl_ms            = get_stats(lpips_mse_list)

            has_mse = model_mse is not None

            mse_psnr_row  = f"| PSNR   | UNet-MSE    | {mp_ms:>8.2f} | {sp_ms:>10.2f} | {mnp_ms:>6.2f} | {mxp_ms:>6.2f} |" if has_mse else "| PSNR   | UNet-MSE    |   N/A    |     N/A    |   N/A  |   N/A  |"
            mse_ssim_row  = f"| SSIM   | UNet-MSE    | {ms_ms:>8.4f} | {ss_ms:>10.4f} | {mns_ms:>6.4f} | {mxs_ms:>6.4f} |" if has_mse else "| SSIM   | UNet-MSE    |   N/A    |     N/A    |   N/A  |   N/A  |"
            mse_lpips_row = f"| LPIPS  | UNet-MSE    | {ml_ms:>8.4f} | {sl_ms:>10.4f} | {mnl_ms:>6.4f} | {mxl_ms:>6.4f} |" if has_mse else "| LPIPS  | UNet-MSE    |   N/A    |     N/A    |   N/A  |   N/A  |"

            delta_mse_vs_base = (
                f"  Δ PSNR : {mp_ms - mp_hx:+.2f}\n"
                f"  Δ SSIM : {ms_ms - ms_hx:+.4f}\n"
                f"  Δ LPIPS: {ml_ms - ml_hx:+.4f} (Negativo es mejor)"
            ) if has_mse else "  UNet-MSE no disponible para esta amplitud."

            delta_prop_vs_mse = (
                f"  Δ PSNR : {mp_pr - mp_ms:+.2f}\n"
                f"  Δ SSIM : {ms_pr - ms_ms:+.4f}\n"
                f"  Δ LPIPS: {ml_pr - ml_ms:+.4f} (Negativo es mejor)"
            ) if has_mse else "  UNet-MSE no disponible para comparar."

            report = f"""Resultados Amplitud {amplitud} - DATASET: {dname}
=========================================================
| Métrica | Método      | Promedio | Desv. Est. | Mínimo | Máximo |
|---------|-------------|----------|------------|--------|--------|
| PSNR   | Baseline    | {mp_hx:>8.2f} | {sp_hx:>10.2f} | {mnp_hx:>6.2f} | {mxp_hx:>6.2f} |
| PSNR   | Wiener      | {mp_wi:>8.2f} | {sp_wi:>10.2f} | {mnp_wi:>6.2f} | {mxp_wi:>6.2f} |
{mse_psnr_row}
| PSNR   | UNet-Prop   | {mp_pr:>8.2f} | {sp_pr:>10.2f} | {mnp_pr:>6.2f} | {mxp_pr:>6.2f} |
| SSIM   | Baseline    | {ms_hx:>8.4f} | {ss_hx:>10.4f} | {mns_hx:>6.4f} | {mxs_hx:>6.4f} |
| SSIM   | Wiener      | {ms_wi:>8.4f} | {ss_wi:>10.4f} | {mns_wi:>6.4f} | {mxs_wi:>6.4f} |
{mse_ssim_row}
| SSIM   | UNet-Prop   | {ms_pr:>8.4f} | {ss_pr:>10.4f} | {mns_pr:>6.4f} | {mxs_pr:>6.4f} |
| LPIPS  | Baseline    | {ml_hx:>8.4f} | {sl_hx:>10.4f} | {mnl_hx:>6.4f} | {mxl_hx:>6.4f} |
| LPIPS  | Wiener      | {ml_wi:>8.4f} | {sl_wi:>10.4f} | {mnl_wi:>6.4f} | {mxl_wi:>6.4f} |
{mse_lpips_row}
| LPIPS  | UNet-Prop   | {ml_pr:>8.4f} | {sl_pr:>10.4f} | {mnl_pr:>6.4f} | {mxl_pr:>6.4f} |

Mejora UNet-Prop vs Baseline (Δ):
  Δ PSNR : {mp_pr - mp_hx:+.2f}
  Δ SSIM : {ms_pr - ms_hx:+.4f}
  Δ LPIPS: {ml_pr - ml_hx:+.4f} (Negativo es mejor)

Mejora UNet-Prop vs Wiener (Δ):
  Δ PSNR : {mp_pr - mp_wi:+.2f}
  Δ SSIM : {ms_pr - ms_wi:+.4f}
  Δ LPIPS: {ml_pr - ml_wi:+.4f} (Negativo es mejor)

Mejora UNet-MSE vs Baseline (Δ):
{delta_mse_vs_base}

Mejora UNet-Prop vs UNet-MSE (Δ):
{delta_prop_vs_mse}
{"-"*70}
"""
            with open(os.path.join(ds_output_dir, "metricas.txt"), "w", encoding="utf-8") as f:
                f.write(report)

            f_summary.write(report + "\n")

print(f"\n[+] Evaluación completada. Revisa la carpeta '{output_root}'.")



