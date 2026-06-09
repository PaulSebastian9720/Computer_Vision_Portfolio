#!/bin/bash
# Descarga los modelos ONNX de MediaPipe sin necesitar la librería Python.
# Fuente: PINTO0309/PINTO_model_zoo (conversiones TFLite → ONNX de los modelos oficiales)
# GitHub: https://github.com/PINTO0309/PINTO_model_zoo

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

echo "=== Descargando palm_detection_lite (128x128, 896 anchors) ==="
wget -q --show-progress \
  "https://github.com/PINTO0309/PINTO_model_zoo/raw/main/033_Hand_Detection_and_Tracking/01_palm_detection_lite/model_float32.onnx" \
  -O palm_detection.onnx

echo "=== Descargando hand_landmark_lite (224x224, 21 keypoints) ==="
wget -q --show-progress \
  "https://github.com/PINTO0309/PINTO_model_zoo/raw/main/033_Hand_Detection_and_Tracking/02_hand_landmark_lite/model_float32.onnx" \
  -O hand_landmark.onnx

echo ""
echo "Listo. Modelos guardados:"
echo "  models/palm_detection.onnx"
echo "  models/hand_landmark.onnx"
echo ""
echo "Si los links no funcionan, busca en:"
echo "  https://github.com/PINTO0309/PINTO_model_zoo"
echo "  carpeta 033_Hand_Detection_and_Tracking"
