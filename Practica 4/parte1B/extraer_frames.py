# Practica 4 - Parte 1B(b): extrae 10 frames distintos (repartidos en todo el video) y los
# guarda como imagenes. Solo extraccion, no aplica super-resolucion (eso lo hace
# mejorar_frames.py aparte, sobre esta misma carpeta).

from pathlib import Path

import cv2

VIDEO_PATH = Path(__file__).parent.parent / "artifacts" / "video" / "La Gran Entrevista _ Mickey Mouse.mp4"
OUT_DIR = Path(__file__).parent / "frames_originales"
N_FRAMES = 10

OUT_DIR.mkdir(exist_ok=True)

cap = cv2.VideoCapture(str(VIDEO_PATH))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
indices = [int(total * i / N_FRAMES) for i in range(N_FRAMES)]

for i, idx in enumerate(indices):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if not ret:
        continue
    out_path = OUT_DIR / f"frame_{i:02d}.jpg"
    cv2.imwrite(str(out_path), frame)
    print("Guardado:", out_path)

cap.release()
print(f"\n{N_FRAMES} frames guardados en {OUT_DIR}")
