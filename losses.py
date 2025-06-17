import torch
import torch.nn.functional as F

# Definición de la función de pérdida
loss_fn = torch.nn.MSELoss()

# Gradientes Sobel
def sobel_gradients(img):
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=img.dtype, device=img.device).view(1,1,3,3)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=img.dtype, device=img.device).view(1,1,3,3)
    grad_x = F.conv2d(img, sobel_x, padding=1, groups=img.shape[1])
    grad_y = F.conv2d(img, sobel_y, padding=1, groups=img.shape[1])
    return grad_x, grad_y

def loss_unet_x(unet, h, x, apply_filter):
    return loss_fn(h(unet(x)), x)

def loss_unet_x_high_freq(unet, h, x, lambd=0.1):
    recon = h(unet(x))
    # Primer término: MSE
    recon_loss = F.mse_loss(recon, x)
    # Gradientes
    grad_x_recon, grad_y_recon = sobel_gradients(recon)
    grad_x, grad_y = sobel_gradients(x)
    # Segundo término: L1 de gradientes
    grad_loss = F.l1_loss(grad_x_recon, grad_x) + F.l1_loss(grad_y_recon, grad_y)
    return recon_loss + lambd * grad_loss

def loss_unet_wiener(unet, h, x, apply_filter):
    Iwiener = apply_filter(x, 'wiener')
    return loss_fn(h(unet(Iwiener)), x)

def loss_unet_media(unet, h, x, apply_filter):
    Imedia = apply_filter(x, 'median')
    return loss_fn(h(unet(Imedia)), x)

def loss_cnn0_x(cnn0, h, x, apply_filter):
    return loss_fn(h(cnn0(x)), x)

def loss_cnn0_wiener(cnn0, h, x, apply_filter):
    Iwiener = apply_filter(x, 'wiener')
    return loss_fn(h(cnn0(Iwiener)), x)

def loss_cnn0_media(cnn0, h, x, apply_filter):
    Imedia = apply_filter(x, 'median')
    return loss_fn(h(cnn0(Imedia)), x)

def loss_wiener_unet_x(unet, h, x, apply_filter):
    Iwiener = apply_filter(x, 'wiener')
    return loss_fn(h(Iwiener + unet(x)), x)

def loss_wiener_unet_wiener(unet, h, x, apply_filter):
    Iwiener = apply_filter(x, 'wiener')
    return loss_fn(h(Iwiener + unet(Iwiener)), x)

def loss_wiener_cnn0_x(cnn0, h, x, apply_filter):
    Iwiener = apply_filter(x, 'wiener')
    return loss_fn(h(Iwiener + cnn0(x)), x)

def loss_wiener_cnn0_wiener(cnn0, h, x, apply_filter):
    Iwiener = apply_filter(x, 'wiener')
    return loss_fn(h(Iwiener + cnn0(Iwiener)), x)

def loss_cnn0_unet_x(unet, cnn0, h, x, apply_filter):
    return loss_fn(h(unet(x) + cnn0(x)), x)

def loss_cnn0_unet_wiener(unet, cnn0, h, x, apply_filter):
    Iwiener = apply_filter(x, 'wiener')
    return loss_fn(h(unet(Iwiener) + cnn0(Iwiener)), x)

def loss_cnn0_x_constraint(cnn0, h, x, apply_filter, alpha=1.0):
    y = h(x)
    x_hat = cnn0(y)
    recon_loss = F.mse_loss(x_hat, x)
    constraint_loss = F.mse_loss(h(x_hat), y)
    return recon_loss + alpha * constraint_loss

def loss_unet_x_constraint(unet, h, x, apply_filter, alpha=1.0):
    y = h(x)
    x_hat = unet(y)
    recon_loss = F.mse_loss(x_hat, x)
    constraint_loss = F.mse_loss(h(x_hat), y)
    return recon_loss + alpha * constraint_loss

# Diccionario de funciones de pérdida
loss_functions = {
    "unet_x": loss_unet_x,
    "unet_wiener": loss_unet_wiener,
    "unet_media": loss_unet_media,
    "cnn0_x": loss_cnn0_x,
    "cnn0_wiener": loss_cnn0_wiener,
    "cnn0_media": loss_cnn0_media,
    "wiener_unet_x": loss_wiener_unet_x,
    "wiener_unet_wiener": loss_wiener_unet_wiener,
    "wiener_cnn0_x": loss_wiener_cnn0_x,
    "wiener_cnn0_wiener": loss_wiener_cnn0_wiener,
    "cnn0_unet_x": loss_cnn0_unet_x,
    "cnn0_unet_wiener": loss_cnn0_unet_wiener,
    "unet_x_high_freq": loss_unet_x_high_freq,
    "cnn0_x_constraint": loss_cnn0_x_constraint,
    "unet_x_constraint": loss_unet_x_constraint,
}