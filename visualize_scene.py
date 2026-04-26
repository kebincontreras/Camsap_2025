"""
visualize_scene.py
==================
Genera 4 imágenes en RGB para una escena seleccionada por el usuario.

Pipeline de pre-corrección (paradigma correcto del sistema):
  1. X            → imagen original (GT)
  2. H·X          → lo que el miope ve SIN corrección (baseline)
  3. f(X)         → imagen pre-corregida por el UNet (lo que se muestra en pantalla)
  4. H·f(X)       → lo que el miope percibe CON corrección (debe ≈ X)

Salida: visuales/<nombre_escena>/
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision.transforms import ToTensor, Resize

# ── Modelo y utilidades del proyecto ──────────────────────────────────────────
from Resources.Ultris.Ultris_model import UNet
from Resources.Ultris.Ultris_zernike import generate_zernike_map, generate_psf, device

# =============================================================================
# CONFIGURACIÓN  (ajusta según tus pesos / amplitud deseada)
# =============================================================================
AMPLITUDE   = 3.0                              # Amplitud de la aberración (miopía)
N, M        = 2, 0                             # Zernike: desenfoque esférico
WEIGHTS_DIR = "Resources/weights_prop_UNET"   # Carpeta con los .pt entrenados
OUTPUT_ROOT = "visuales"                       # Carpeta raíz de salida
RESIZE_TO   = (512, 512)                       # Tamaño al que se reescala la imagen
CROP        = 20                               # Píxeles a recortar en bordes

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def get_weight_path(base_dir: str, amplitud: float) -> str | None:
    """Busca el archivo de pesos con fallback para nombres sin decimales."""
    candidates = [
        os.path.join(base_dir, f"modelo_final_{amplitud}.pt"),
        os.path.join(base_dir, f"modelo_final_{int(amplitud)}.pt"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def load_unet(weight_path: str) -> UNet:
    """Carga el UNet en modo evaluación."""
    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    return model


def load_image_rgb(path: str) -> np.ndarray:
    """Carga imagen como array RGB float32 en [0, 1], redimensionada a RESIZE_TO."""
    img = Image.open(path).convert("RGB")
    img = img.resize((RESIZE_TO[1], RESIZE_TO[0]), Image.LANCZOS)
    return np.array(img).astype(np.float32) / 255.0   # [H, W, 3]


def apply_psf_channel(channel_np: np.ndarray, psf_tensor: torch.Tensor) -> np.ndarray:
    """Aplica la PSF (convolución) a un canal 2D usando F.conv2d."""
    t = torch.tensor(channel_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    out = F.conv2d(t, psf_tensor, padding="same")
    return out.squeeze().cpu().numpy()


def apply_psf_rgb(img_np: np.ndarray, psf_tensor: torch.Tensor) -> np.ndarray:
    """Aplica la PSF a cada canal RGB por separado."""
    out = np.zeros_like(img_np)
    for c in range(3):
        out[..., c] = apply_psf_channel(img_np[..., c], psf_tensor)
    return np.clip(out, 0.0, 1.0)


def run_unet_rgb(img_np: np.ndarray, model: UNet, psf_tensor: torch.Tensor) -> np.ndarray:
    """
    Pasa cada canal por el UNet (que opera en escala de grises, 1 canal)
    y reconstruye la imagen RGB de salida.
    """
    # El UNet fue entrenado con h*x (imagen aberrada) como entrada
    # Primero aplicamos la PSF para obtener la imagen aberrada
    aberrated_np = apply_psf_rgb(img_np, psf_tensor)  # H * X en RGB

    out = np.zeros_like(img_np)
    for c in range(3):
        ch = aberrated_np[..., c]
        t  = torch.tensor(ch, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(t)
        out[..., c] = pred.squeeze().cpu().numpy()
    return np.clip(out, 0.0, 1.0)


def run_unet_single_channel(channel_np: np.ndarray, model: UNet) -> np.ndarray:
    """Pasa un canal 2D por el UNet y devuelve la salida como array 2D."""
    t = torch.tensor(channel_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(t)
    return np.clip(pred.squeeze().cpu().numpy(), 0.0, 1.0)


def crop_center(img_np: np.ndarray) -> np.ndarray:
    """Recorta los bordes de la imagen."""
    return img_np[CROP:-CROP, CROP:-CROP]


def save_rgb(img_np: np.ndarray, path: str):
    """Guarda un array float32 [H, W, 3] como PNG en 8 bits."""
    out = (np.clip(img_np, 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(out, mode="RGB").save(path)
    print(f"  ✔ Guardada: {path}")


# =============================================================================
# SELECCIÓN INTERACTIVA DE LA ESCENA
# =============================================================================

def select_scene() -> tuple[str, str]:
    """
    Pide al usuario la ruta de la imagen y devuelve (ruta_imagen, nombre_escena).
    Acepta:
      - Ruta directa a un archivo de imagen.
      - Ruta a una carpeta (toma la primera imagen .png/.jpg encontrada).
    """
    print("\n" + "="*60)
    print("  VISUALIZADOR DE ESCENAS — Proyecto CAMSAP 2025")
    print("="*60)
    print("\nPuedes ingresar:")
    print("  • La ruta completa a un archivo de imagen (.png / .jpg / .bmp)")
    print("  • La ruta a una carpeta (se usará la 1ª imagen encontrada)")
    print()

    while True:
        ruta = input("📂 Ingresa la ruta de la escena: ").strip().strip('"').strip("'")

        if os.path.isfile(ruta):
            ext = os.path.splitext(ruta)[1].lower()
            if ext in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}:
                nombre_escena = os.path.splitext(os.path.basename(ruta))[0]
                return ruta, nombre_escena
            else:
                print(f"  ✘ Extensión '{ext}' no soportada. Usa .png, .jpg, .bmp, etc.\n")

        elif os.path.isdir(ruta):
            imagenes = sorted([
                f for f in os.listdir(ruta)
                if os.path.splitext(f)[1].lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
            ])
            if not imagenes:
                print(f"  ✘ No se encontraron imágenes en '{ruta}'.\n")
            else:
                archivo = os.path.join(ruta, imagenes[0])
                print(f"  → Usando imagen: {imagenes[0]}")
                nombre_escena = os.path.basename(os.path.normpath(ruta))
                return archivo, nombre_escena

        else:
            print(f"  ✘ Ruta no válida: '{ruta}'\n")


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def main():
    # ── 1. Selección de escena ─────────────────────────────────────────────────
    image_path, scene_name = select_scene()
    print(f"\n✅ Escena seleccionada: '{scene_name}'")
    print(f"   Archivo: {image_path}")

    # ── 2. Carpeta de salida ───────────────────────────────────────────────────
    out_dir = os.path.join(OUTPUT_ROOT, scene_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"   Carpeta de salida: {out_dir}\n")

    # ── 3. Cargar modelo ───────────────────────────────────────────────────────
    wp = get_weight_path(WEIGHTS_DIR, AMPLITUDE)
    if wp is None:
        # Fallback: busca en la carpeta Resources/Ultris (modelo de ejemplo)
        wp_fallback = os.path.join("Resources", "Ultris", f"modelo_final_{int(AMPLITUDE)}.pt")
        if os.path.exists(wp_fallback):
            wp = wp_fallback
            print(f"⚠ Usando pesos de fallback: {wp_fallback}")
        else:
            sys.exit(
                f"✘ No se encontraron pesos para amplitud={AMPLITUDE} en '{WEIGHTS_DIR}'.\n"
                f"  Verifica la ruta o ajusta WEIGHTS_DIR / AMPLITUDE en el script."
            )
    print(f"[+] Cargando modelo desde: {wp}")
    model = load_unet(wp)
    print(f"    → Dispositivo: {device}\n")

    # ── 4. Generar PSF ─────────────────────────────────────────────────────────
    print("[+] Generando PSF (Zernike miopía)...")
    zmap       = generate_zernike_map(N, M, amplitude=AMPLITUDE)
    psf        = generate_psf(zmap)                                      # [256, 256]
    psf_tensor = psf.unsqueeze(0).unsqueeze(0).to(device)               # [1,1,256,256]
    print(f"    PSF shape: {psf_tensor.shape}\n")

    # ── 5. Cargar imagen ───────────────────────────────────────────────────────
    print("[+] Cargando imagen...")
    x_rgb = load_image_rgb(image_path)   # [H, W, 3]  float32 [0,1]
    print(f"    Dimensiones: {x_rgb.shape}\n")

    # ── 6. Calcular las 4 imágenes (pipeline de pre-corrección) ───────────────
    # Imagen 2: H·X  — lo que el miope ve sin ninguna corrección
    print("[+] Aplicando PSF a X  →  H·X  (miope sin corrección)...")
    hx_rgb = apply_psf_rgb(x_rgb, psf_tensor)

    # Imagen 3: f(X)  — UNet transforma X para mostrarlo en pantalla pre-corregido
    print("[+] Pasando X por el UNet  →  f(X)  (imagen pre-corregida para pantalla)...")
    fx_rgb = np.zeros_like(x_rgb)
    for c in range(3):
        fx_rgb[..., c] = run_unet_single_channel(x_rgb[..., c], model)
    fx_rgb = np.clip(fx_rgb, 0.0, 1.0)

    # Imagen 4: H·f(X)  — lo que el miope percibe al ver f(X) en pantalla (debe ≈ X)
    print("[+] Aplicando PSF a f(X)  →  H·f(X)  (percepción del miope con corrección)...")
    hfx_rgb = apply_psf_rgb(fx_rgb, psf_tensor)

    # Recortar bordes para eliminar artefactos de convolución
    x_out   = crop_center(x_rgb)
    hx_out  = crop_center(hx_rgb)
    fx_out  = crop_center(fx_rgb)
    hfx_out = crop_center(hfx_rgb)

    # ── 7. Guardar imágenes ────────────────────────────────────────────────────
    print(f"\n[+] Guardando imágenes en '{out_dir}'...")
    save_rgb(x_out,   os.path.join(out_dir, "1_x_original.png"))
    save_rgb(hx_out,  os.path.join(out_dir, "2_hx_sin_correccion.png"))
    save_rgb(fx_out,  os.path.join(out_dir, "3_fx_precorregida.png"))
    save_rgb(hfx_out, os.path.join(out_dir, "4_hfx_percepcion.png"))

    print(f"\n{'='*60}")
    print(f"  ✅ Listo. 4 imágenes guardadas en:  {out_dir}")
    print(f"{'='*60}\n")
    print("  Imágenes generadas:")
    print(f"    1_x_original.png       →  X         (GT — imagen original)")
    print(f"    2_hx_sin_correccion.png →  H·X      (miope SIN corrección)")
    print(f"    3_fx_precorregida.png  →  f(X)      (lo que se muestra en pantalla)")
    print(f"    4_hfx_percepcion.png   →  H·f(X)   (lo que el miope percibe, debe ≈ X)")
    print()


if __name__ == "__main__":
    main()
