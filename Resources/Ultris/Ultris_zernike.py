try:
    import torch
    xp = torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    GPU_ENABLED = torch.cuda.is_available()
except ImportError:
    import numpy as np
    xp = np
    device = "cpu"
    GPU_ENABLED = False

if GPU_ENABLED:
    print("🚀 Ejecutando con GPU (PyTorch)")
else:
    print("⚙️ Ejecutando con CPU (NumPy)")


import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve

# Diccionario de aberraciones comunes con sus (n, m)
aberrations = {
    "Sin aberración": (0, 0),
    "Inclinación vertical (Tilt)": (1, -1),
    "Inclinación horizontal (Tilt)": (1, 1),
    "Desenfoque esférico (Miopía)": (2, 0),
    "Astigmatismo oblicuo": (2, -2),
    "Astigmatismo vertical": (2, 2),
    "Coma vertical": (3, -1),
    "Coma horizontal": (3, 1),
    "Trefoil oblicuo": (3, -3),
    "Trefoil vertical": (3, 3),
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{'🚀 GPU' if torch.cuda.is_available() else '⚙️ CPU'} activado con PyTorch")

def zernike_radial(n, m, rho):
    R = torch.zeros_like(rho)
    m = abs(m)
    for k in range((n - m) // 2 + 1):
        coef = (-1)**k * torch.lgamma(torch.tensor(n - k + 1)) \
            - torch.lgamma(torch.tensor(k + 1)) \
            - torch.lgamma(torch.tensor((n + m) // 2 - k + 1)) \
            - torch.lgamma(torch.tensor((n - m) // 2 - k + 1))
        coef = torch.exp(coef)
        print(f"[DEBUG] Coeficiente en k={k}: {coef}")
        R += coef * rho**(n - 2 * k)
        print(f"[DEBUG] R acumulado en k={k}: {R}")
    return R

def zernike(n, m, rho, theta):
    R = zernike_radial(n, m, rho)
    if m > 0:
        return R * torch.cos(m * theta)
    elif m < 0:
        return R * torch.sin(-m * theta)
    else:
        return R

def generate_zernike_map(n, m, size=256, amplitude=1.0):
    y = torch.linspace(-1, 1, size, device=device)
    x = torch.linspace(-1, 1, size, device=device)
    X, Y = torch.meshgrid(y, x, indexing='ij')
    rho = torch.sqrt(X**2 + Y**2)
    theta = torch.atan2(Y, X)
    print(f"[DEBUG] Valores de rho: {rho}")
    print(f"[DEBUG] Valores de theta: {theta}")
    mask = rho <= 1
    print(f"[DEBUG] Máscara aplicada: {mask}")
    Z = torch.zeros_like(rho)
    Z[mask] = amplitude * zernike(n, m, rho[mask], theta[mask])
    print(f"[DEBUG] Zernike map generado con n={n}, m={m}, amplitud={amplitude}: {Z.shape}, valores: {Z}")
    return Z

#def generate_psf(zernike_map):
#    pupil_function = torch.exp(1j * 2 * torch.pi * zernike_map)
#    fft = torch.fft.fft2(pupil_function)
#    psf = torch.fft.fftshift(torch.abs(fft) ** 2)
#    psf = psf / psf.sum()
#    return psf.real  # usamos solo la parte real (ya es real)

def generate_psf(zernike_map):
    if torch.all(zernike_map == 0):
        # Crear un delta en el centro (PSF ideal sin aberración)
        size = zernike_map.shape[0]
        psf = torch.zeros_like(zernike_map)
        psf[size // 2, size // 2] = 1.0
        print(f"[DEBUG] PSF ideal generado (sin aberración): {psf.shape}, valores: {psf}")
        return psf
    else:
        pupil_function = torch.exp(1j * 2 * torch.pi * zernike_map)
        fft = torch.fft.fft2(pupil_function)
        psf = torch.fft.fftshift(torch.abs(fft) ** 2)
        psf = psf / psf.sum()
        print(f"[DEBUG] PSF generado: {psf.shape}, valores: {psf}")
        return psf.real  # usamos solo la parte real (ya es real)


def apply_psf(image, psf):
    # Convertimos a NumPy porque scipy.ndimage.convolve requiere NumPy
    psf_np = psf.detach().cpu().numpy()
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()

    output = np.zeros_like(image)
    for c in range(3):
        output[..., c] = convolve(image[..., c], psf_np, mode='reflect')
    return output
