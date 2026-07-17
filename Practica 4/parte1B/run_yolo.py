# Practica 4 - Parte 1B(a): YOLOv12 / YOLOv26 sobre el video de artifacts/.
# Cambiar MODEL_NAME / DEVICE y correr una vez por combinacion. 'q' para salir.

import os

# En sesion Wayland, el Qt que trae opencv-python solo incluye el plugin "xcb" (X11), no
# "wayland". Sin esto, cv2.imshow no crea ninguna ventana visible (falla en silencio).
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import time
from pathlib import Path

import cv2
from ultralytics import YOLO

MODEL_NAME = "yolo12n"  # "yolo12n" o "yolo26n"
DEVICE = "cpu"           # "cuda" o "cpu"

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "modelos" / f"{MODEL_NAME}.pt"
VIDEO_PATH = BASE_DIR.parent / "artifacts" / "video" / "personas.mp4"

model = YOLO(str(MODEL_PATH))
cap = cv2.VideoCapture(str(VIDEO_PATH))

prev_t = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, device=DEVICE, conf=0.4, verbose=False)
    annotated = results[0].plot()

    now = time.time()
    fps = 1.0 / max(now - prev_t, 1e-6)
    prev_t = now
    cv2.putText(annotated, f"{MODEL_NAME} - {DEVICE.upper()} - FPS: {fps:.1f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(annotated, f"{MODEL_NAME} - {DEVICE.upper()} - FPS: {fps:.1f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)

    cv2.imshow(f"{MODEL_NAME} - {DEVICE}", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
