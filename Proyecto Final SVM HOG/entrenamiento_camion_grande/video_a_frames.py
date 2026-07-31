"""Convierte el video de la webcam (grabado sin ningún camión en escena) en frames
JPG dentro de artifcats/webcam_negatives/, para usarlos como fuente de
hard-negative mining en 02_entrenamiento_v2.ipynb.
"""
from pathlib import Path

import cv2

VIDEO_PATH = Path(__file__).resolve().parent.parent / "artifcats" / "webcam_negativos.webm"
OUT_DIR = Path(__file__).resolve().parent.parent / "artifcats" / "webcam_negatives"
FRAME_STEP = 3  # guarda 1 de cada 3 frames (frames consecutivos son casi idénticos)

OUT_DIR.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(VIDEO_PATH))
if not cap.isOpened():
    raise SystemExit(f"No se pudo abrir {VIDEO_PATH}")

n_read, n_saved = 0, 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if n_read % FRAME_STEP == 0:
        cv2.imwrite(str(OUT_DIR / f"neg_{n_saved:04d}.jpg"), frame)
        n_saved += 1
    n_read += 1

cap.release()
print(f"video: {n_read} frames leídos -> {n_saved} guardados en {OUT_DIR}")
