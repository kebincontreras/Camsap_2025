import torch
import torch.nn.functional as F

loss_fn = torch.nn.MSELoss()

def loss_unet_x(unet, h, x, apply_filter):
    return loss_fn(h(unet(x)), x)

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
}