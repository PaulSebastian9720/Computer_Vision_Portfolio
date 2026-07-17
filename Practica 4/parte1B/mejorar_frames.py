# Practica 4 - Parte 1B(b): aplica Real-ESRGAN a los frames de frames_originales/ y guarda el
# resultado en frames_mejoradas/. Correr extraer_frames.py primero.
# Cambiar DEVICE y correr una vez por combinacion.

from pathlib import Path

import cv2
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

DEVICE = "cpu"     # "cuda" o "cpu"
CROP_SIZE = 160     # mismo recorte central que run_superres.py, por velocidad en CPU

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "modelos" / "RealESRGAN_x4plus.pth"
IN_DIR = BASE_DIR / "frames_originales"
OUT_DIR = BASE_DIR / f"frames_mejoradas_{DEVICE}"
OUT_DIR.mkdir(exist_ok=True)

model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
upsampler = RealESRGANer(
    scale=4,
    model_path=str(MODEL_PATH),
    model=model,
    tile=128,
    half=(DEVICE == "cuda"),
    device=torch.device(DEVICE),
)


def center_crop(frame, size):
    h, w = frame.shape[:2]
    s = min(size, h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    return frame[y0:y0 + s, x0:x0 + s]


for img_path in sorted(IN_DIR.glob("*.jpg")):
    frame = cv2.imread(str(img_path))
    crop = center_crop(frame, CROP_SIZE)
    mejorado, _ = upsampler.enhance(crop, outscale=4)

    out_path = OUT_DIR / img_path.name
    cv2.imwrite(str(out_path), mejorado)
    print("Guardado:", out_path)

print(f"\nListo. Imagenes mejoradas en {OUT_DIR}")
