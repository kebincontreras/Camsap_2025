import gradio as gr
import torch
import numpy as np
import cv2
from torchvision.transforms import ToTensor, Resize, Grayscale
from Resources.Ultris.Ultris_model import UNet
import torch.nn.functional as F

# Configuración
n, m = 2, 0
IMG_SIZE = 256
AMP = 3.0
WEIGHT_PATH = "Resources/weights/modelo_final_1.pt"  # Sube este archivo a Hugging Face Spaces

resize_to_256 = Resize((IMG_SIZE, IMG_SIZE))

def load_model(weight_path):
    device = torch.device('cpu')
    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    return model, device

model, device = load_model(WEIGHT_PATH)

def generate_zernike_map_local(n, m, size=256, amplitude=1.0):
    y = torch.linspace(-1, 1, size, device=device)
    x = torch.linspace(-1, 1, size, device=device)
    X, Y = torch.meshgrid(y, x, indexing='ij')
    rho = torch.sqrt(X**2 + Y**2)
    theta = torch.atan2(Y, X)
    mask = rho <= 1
    Z = torch.zeros_like(rho)
    Z[mask] = amplitude * zernike_local(n, m, rho[mask], theta[mask])
    return Z

def zernike_local(n, m, rho, theta):
    R = torch.zeros_like(rho)
    m = abs(m)
    for k in range((n - m) // 2 + 1):
        coef = (-1)**k * torch.lgamma(torch.tensor(n - k + 1)) \
            - torch.lgamma(torch.tensor(k + 1)) \
            - torch.lgamma(torch.tensor((n + m) // 2 - k + 1)) \
            - torch.lgamma(torch.tensor((n - m) // 2 - k + 1))
        coef = torch.exp(coef)
        R += coef * rho**(n - 2 * k)
    if m > 0:
        return R * torch.cos(m * theta)
    elif m < 0:
        return R * torch.sin(-m * theta)
    else:
        return R

def generate_psf_local(zernike_map):
    if torch.all(zernike_map == 0):
        size = zernike_map.shape[0]
        psf = torch.zeros_like(zernike_map)
        psf[size // 2, size // 2] = 1.0
        return psf
    else:
        pupil_function = torch.exp(1j * 2 * torch.pi * zernike_map)
        fft = torch.fft.fft2(pupil_function)
        psf = torch.fft.fftshift(torch.abs(fft) ** 2)
        psf = psf / psf.sum()
        return psf.real

# Precompute PSF and its FFT
zmap = generate_zernike_map_local(n, m, amplitude=AMP)
psf = generate_psf_local(zmap)
psf_tensor = psf.unsqueeze(0).unsqueeze(0).to(device)
if psf_tensor.shape[-1] != IMG_SIZE or psf_tensor.shape[-2] != IMG_SIZE:
    psf_tensor = torch.nn.functional.interpolate(psf_tensor, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
psf_tensor = psf_tensor / psf_tensor.sum()
psf_tensor = torch.fft.fftshift(psf_tensor, dim=(-2, -1))
psf_fft = torch.fft.fft2(psf_tensor)

def aberrate_image_fft(img_tensor, psf_fft):
    print(f"[DEBUG] Dimensiones del tensor de entrada para aberración: {img_tensor.shape}")
    img_fft = torch.fft.fft2(img_tensor)
    print(f"[DEBUG] Dimensiones del tensor FFT de la imagen: {img_fft.shape}")
    result_fft = img_fft * psf_fft
    print(f"[DEBUG] Dimensiones del tensor FFT del resultado: {result_fft.shape}")
    result = torch.fft.ifft2(result_fft).real
    print(f"[DEBUG] Dimensiones del tensor de la imagen aberrada: {result.shape}")
    return result

def restore_image(model, aberrated_tensor):
    with torch.no_grad():
        restored = model(aberrated_tensor)
    return restored.squeeze().cpu().numpy()

def process_image(image):
    if image is None:
        print("[DEBUG] Input image is None.")
        return None, None

    # Validate input image format
    if not isinstance(image, np.ndarray):
        raise ValueError("Input must be a NumPy image.")

    print(f"[DEBUG] Input image dimensions: {image.shape}")

    # Validate color image
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Input must be a color image (3 channels).")

    # Resize image to 256x256
    img_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    print(f"[DEBUG] Resized image dimensions: {img_resized.shape}")

    img_np = img_resized.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float().to(device)  # [1, C, H, W]

    print(f"[DEBUG] Image tensor dimensions: {img_tensor.shape}")

    with torch.no_grad():
        # Process each channel separately
        aberrated_channels = []
        restored_channels = []
        for c in range(img_tensor.shape[1]):
            channel_tensor = img_tensor[:, c:c+1, :, :]
            aberrated_tensor = aberrate_image_fft(channel_tensor, psf_fft)
            restored_tensor = model(aberrated_tensor)
            aberrated_channels.append(aberrated_tensor.squeeze(0).cpu().numpy())
            restored_channels.append(restored_tensor.squeeze(0).cpu().numpy())

        # Combine aberrated channels
        aberrated_np = np.stack(aberrated_channels, axis=-1)
        aberrated_np = np.clip(aberrated_np, 0, 1)

        # Combine restored channels
        restored_np = np.stack(restored_channels, axis=-1)
        restored_np = np.clip(restored_np, 0, 1)

    print(f"[DEBUG] Aberrated image dimensions before squeeze: {aberrated_np.shape}")
    print(f"[DEBUG] Restored image dimensions before squeeze: {restored_np.shape}")

    # Remove extra dimensions
    aberrated_np = np.squeeze(aberrated_np)
    restored_np = np.squeeze(restored_np)

    print(f"[DEBUG] Aberrated image dimensions after squeeze: {aberrated_np.shape}")
    print(f"[DEBUG] Restored image dimensions after squeeze: {restored_np.shape}")

    # Convert aberrated image to uint8
    aberrated_resized = (aberrated_np * 255).astype(np.uint8)
    print(f"[DEBUG] Aberrated image dimensions resized: {aberrated_resized.shape}")

    # Convert restored image to uint8
    restored_resized = (restored_np * 255).astype(np.uint8)
    print(f"[DEBUG] Restored image dimensions resized: {restored_resized.shape}")

    return aberrated_resized, restored_resized

def process_images(images):
    if images is None or len(images) == 0:
        return None, None

    processed_images = []
    restored_images = []

    for image in images:
        # Validar que la imagen sea en color
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Cada entrada debe ser una imagen en color (3 canales).")

        # Redimensionar la imagen a 256x256
        img_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        img_np = img_resized.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float().to(device)  # [1, C, H, W]

        with torch.no_grad():
            # Procesar cada canal por separado
            channels = []
            for c in range(img_tensor.shape[1]):
                channel_tensor = img_tensor[:, c:c+1, :, :]
                aberrated_tensor = aberrate_image_fft(channel_tensor, psf_fft)
                restored_tensor = model(aberrated_tensor)
                channels.append(restored_tensor.squeeze(0).cpu().numpy())

            # Combinar canales restaurados
            restored_np = np.stack(channels, axis=-1)
            restored_np = np.clip(restored_np, 0, 1)

        # Convertir la imagen restaurada a formato uint8
        restored_resized = (restored_np * 255).astype(np.uint8)

        # Asegurar que las imágenes devueltas sean del formato esperado
        img_resized = img_resized.astype(np.uint8)

        processed_images.append(img_resized)
        restored_images.append(restored_resized)

    return processed_images, restored_images

# Actualizar el título dinámicamente para reflejar GPU o CPU
device_type = 'GPU' if torch.cuda.is_available() else 'CPU'
with gr.Blocks() as demo:
    gr.Markdown(f"# Restauración de Imágenes con Aberración de Zernike y UNet ({device_type}, Hugging Face Spaces)")
    with gr.Row():
        input_img = gr.Image(label="Imagen de entrada", type="numpy")
        aberrated = gr.Image(label="Imagen aberrada")
        restored = gr.Image(label="Imagen transformada")
    btn = gr.Button("Procesar")
    btn.click(fn=process_image, inputs=[input_img], outputs=[aberrated, restored])

demo.launch()
