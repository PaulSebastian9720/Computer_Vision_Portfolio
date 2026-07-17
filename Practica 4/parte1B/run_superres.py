# Practica 4 - Parte 1B(b): Real-ESRGAN sobre el video de artifacts/.
# Cambiar DEVICE y correr una vez por combinacion. 'q' para salir.
#
# Se procesa un recorte central pequeno (CROP_SIZE) en vez del frame completo: a resolucion
# completa, Real-ESRGAN en CPU tarda minutos por frame (854x480 -> 28 tiles), inviable para ver
# algo en vivo. Con el recorte se ve fluido y alcanza para comparar CPU vs GPU.
# tile=128 se deja fijo: sin el, la GPU (MX230, ~2GB VRAM) se queda sin memoria.
#
# La ventana muestra lado a lado: recorte original (interpolado al mismo tamano de salida, para
# que la comparacion sea justa) | resultado de Real-ESRGAN.

import os

# En sesion Wayland, el Qt que trae opencv-python solo incluye el plugin "xcb" (X11), no
# "wayland". Sin esto, cv2.imshow no crea ninguna ventana visible (falla en silencio).
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import time
from pathlib import Path

import cv2
import numpy as np
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

DEVICE = "cpu"     # "cuda" o "cpu"
CROP_SIZE = 160     # lado del recorte central (px) que se envia a la red, antes del x4

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "modelos" / "RealESRGAN_x4plus.pth"
VIDEO_PATH = BASE_DIR.parent / "artifacts" / "video" / "La Gran Entrevista _ Mickey Mouse.mp4"

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


cap = cv2.VideoCapture(str(VIDEO_PATH))

WINDOW_NAME = f"Real-ESRGAN - {DEVICE} (original vs mejorado)"

# Crear y "pintar" la ventana ANTES del primer frame real. Cada frame tarda ~15s en CPU, y con
# solo waitKey(1) por vuelta el compositor (KDE/Wayland vía XWayland) no alcanza a mapear la
# ventana la primera vez -- se queda invisible aunque el proceso siga corriendo. Pumpear eventos
# aqui con una espera mas larga evita eso.
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
placeholder = np.zeros((320, 640, 3), dtype="uint8")
cv2.putText(placeholder, "Cargando primer frame...", (30, 160),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
cv2.imshow(WINDOW_NAME, placeholder)
cv2.waitKey(500)

prev_t = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    crop = center_crop(frame, CROP_SIZE)
    mejorado, _ = upsampler.enhance(crop, outscale=4)

    original_interp = cv2.resize(crop, (mejorado.shape[1], mejorado.shape[0]),
                                  interpolation=cv2.INTER_CUBIC)
    comparacion = cv2.hconcat([original_interp, mejorado])

    now = time.time()
    fps = 1.0 / max(now - prev_t, 1e-6)
    prev_t = now

    cv2.putText(comparacion, "Original (interpolado)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(comparacion, "Original (interpolado)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(comparacion, "Real-ESRGAN", (mejorado.shape[1] + 10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(comparacion, "Real-ESRGAN", (mejorado.shape[1] + 10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(comparacion, f"{DEVICE.upper()} - FPS: {fps:.2f}", (10, comparacion.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(comparacion, f"{DEVICE.upper()} - FPS: {fps:.2f}", (10, comparacion.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

    cv2.imshow(WINDOW_NAME, comparacion)
    if cv2.waitKey(500) & 0xFF == ord("q"):  # 500ms: le da tiempo al compositor de pintar
        break

cap.release()
cv2.destroyAllWindows()
