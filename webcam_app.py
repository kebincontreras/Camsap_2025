
import cv2
import torch
import numpy as np
import time
from Resources.Ultris.Ultris_model import UNet
from Resources.Ultris.Ultris_zernike import generate_zernike_map, generate_psf
import torch.nn.functional as F

# Configuración optimizada
n, m = 2, 0
CROP = 0  # Sin recorte para máxima velocidad
IMG_SIZE = 512  # Mayor calidad visual


if torch.cuda.is_available():
    device = torch.device('cuda')
    print('🚀 Usando GPU (CUDA)')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')
    print('⚙️ Usando CPU')

def crop_center(img):
    if CROP == 0:
        return img
    return img[CROP:-CROP, CROP:-CROP]

def apply_psf_torch(image_tensor, psf_tensor):
    if psf_tensor.max() == 1.0 and psf_tensor.sum() == 1.0 and torch.count_nonzero(psf_tensor) == 1:
        return image_tensor
    return F.conv2d(image_tensor, psf_tensor, padding="same")

def load_model(weight_path):
    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    return model

def aberrate_image_fft(img_tensor, psf_fft):
    # img_tensor: torch tensor [1,1,H,W] en GPU, float32
    # psf_fft: torch tensor [1,1,H,W] en GPU, complex64
    img_fft = torch.fft.fft2(img_tensor)
    result_fft = img_fft * psf_fft
    result = torch.fft.ifft2(result_fft).real
    return result

def restore_image(model, aberrated_tensor):
    with torch.no_grad():
        restored = model(aberrated_tensor)
    return restored.squeeze().cpu().numpy()

# Solo severidad 2.0
WEIGHT_PATH = "Resources/weights/modelo_final_1.pt"
AMP = 1.0

def main():

    model = load_model(WEIGHT_PATH)
    # Precompute PSF and its FFT (fixed for all frames)
    zmap = generate_zernike_map(n, m, amplitude=AMP)
    psf = generate_psf(zmap)
    psf_tensor = psf.unsqueeze(0).unsqueeze(0).to(device)
    # Pad/crop PSF to IMG_SIZE if needed
    if psf_tensor.shape[-1] != IMG_SIZE or psf_tensor.shape[-2] != IMG_SIZE:
        psf_tensor = torch.nn.functional.interpolate(psf_tensor, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
    # Normalize PSF
    psf_tensor = psf_tensor / psf_tensor.sum()
    # Centrar la PSF para FFT (evita artefactos tipo wrap-around)
    psf_tensor = torch.fft.fftshift(psf_tensor, dim=(-2, -1))
    psf_fft = torch.fft.fft2(psf_tensor)

    cap = cv2.VideoCapture(0)
    print("Presiona 'q' para salir.")

    frame_count = 0
    real_disp = aberrated_disp = restored_disp = None
    prev_time = time.time()
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # Procesar solo 1 de cada 5 frames para más fluidez
        if frame_count % 5 == 0:
            t0 = time.time()
            # OpenCV: BGR -> GRAY, resize, normalize
            real_color = cv2.resize(frame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            real_ycrcb = cv2.cvtColor(real_color, cv2.COLOR_BGR2YCrCb)
            real_gray = real_ycrcb[:, :, 0]
            real_norm = real_gray.astype(np.float32) / 255.0
            t1 = time.time()

            # Aberrar (FFT en GPU)
            img_tensor = torch.from_numpy(real_norm).unsqueeze(0).unsqueeze(0).float().to(device)
            with torch.no_grad():
                aberrated_tensor = aberrate_image_fft(img_tensor, psf_fft)
            t2 = time.time()

            # Restaurar
            restored_np = restore_image(model, aberrated_tensor)
            t3 = time.time()

            # Recorte y normalización
            aberrated = crop_center(aberrated_tensor.squeeze().cpu().numpy())
            restored = crop_center(restored_np)
            aberrated = np.clip(aberrated, 0, 1)
            restored = np.clip(restored, 0, 1)
            t4 = time.time()

            # Convertir a formato visualizable en color usando crominancia real
            def gray_to_color(gray_img, ycrcb_ref):
                ycrcb = ycrcb_ref.copy()
                ycrcb[:, :, 0] = (gray_img * 255).astype(np.uint8)
                return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

            aberrated_disp = gray_to_color(aberrated, real_ycrcb)
            restored_disp = gray_to_color(restored, real_ycrcb)
            t5 = time.time()

            # Calcular FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time)
            prev_time = curr_time

            # Debug timings
            print(f"[DEBUG] Preproc: {(t1-t0)*1000:.1f} ms | Aberr: {(t2-t1)*1000:.1f} ms | Restore: {(t3-t2)*1000:.1f} ms | Post: {(t4-t3)*1000:.1f} ms | Color: {(t5-t4)*1000:.1f} ms | Total: {(t5-t0)*1000:.1f} ms")

        # Mostrar el último resultado procesado
        if aberrated_disp is not None and restored_disp is not None:
            combined = np.hstack([aberrated_disp, restored_disp])
            # Mostrar FPS en la imagen
            cv2.putText(combined, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.imshow(f"Aberrada | Restaurada (Severidad: {AMP})", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
