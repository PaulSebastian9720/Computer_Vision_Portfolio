"""Barrido de umbral sobre el modelo YA entrenado (sin reentrenar nada).

Mide, para varios valores de hitThreshold:
  - RECALL: % de camiones reales (positivos usados en entrenamiento) que siguen
    puntuando por encima del umbral.
  - FALSOS POSITIVOS: sobre imágenes de Citypersons (calles densas, el modelo NO
    las ha visto ni en entrenamiento ni en mining todavía) corre detectMultiScale
    completo (igual que la app en C++) y mide en qué % de imágenes aparece al
    menos una detección fantasma.

Objetivo: saber si con el modelo actual alcanza con mover el umbral, antes de
invertir en un reentrenamiento con Citypersons sumado al mining.
"""
import random
from pathlib import Path

import cv2
import numpy as np

random.seed(7)

BASE = Path(__file__).resolve().parent
DATASET_DIR = BASE / "dataset"
ARTIFACTS_DIR = BASE.parent / "artifcats"

VIEWS = {
    "lateral": dict(win_size=(144, 80), yml="detector_lateral.yml",
                     pos_dir=DATASET_DIR / "lateral_positives_raw", stride=(24, 16)),
}

THRESHOLDS = [-0.5, -0.3, -0.1, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.3, 1.5]

# Citypersons: escenas urbanas densas que el modelo NO ha visto en mining todavía
# (el video conflictivo era justo de este tipo de escena)
citypersons_imgs = list((ARTIFACTS_DIR / "Citypersons.coco").rglob("*.png"))
random.shuffle(citypersons_imgs)
test_negatives = citypersons_imgs[:300]
print(f"imagenes de prueba (calles densas, no vistas): {len(test_negatives)}")


def load_detector(cfg):
    fs = cv2.FileStorage(str(DATASET_DIR / cfg["yml"]), cv2.FileStorage_READ)
    detector = fs.getNode("svm_detector").mat()
    fs.release()
    vec = detector.reshape(-1)
    w, b = vec[:-1], vec[-1]

    hog = cv2.HOGDescriptor(cfg["win_size"], (16, 16), (8, 8), (8, 8), 9)
    hog.setSVMDetector(np.append(w, b).astype(np.float32))
    return hog, w, b


results = {}
for name, cfg in VIEWS.items():
    print(f"\n=== vista {name} ===")
    hog, w, b = load_detector(cfg)

    # --- recall: score directo de crops positivos reales (sin sliding window) ---
    pos_paths = sorted(cfg["pos_dir"].glob("*.jpg"))
    pos_sample = random.sample(pos_paths, min(500, len(pos_paths)))
    pos_scores = []
    for p in pos_sample:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, cfg["win_size"])
        feat = hog.compute(img).flatten()
        pos_scores.append(float(np.dot(feat, w) + b))
    pos_scores = np.array(pos_scores)

    # --- falsos positivos: detectMultiScale completo sobre escenas SIN camion ---
    # umbral muy bajo (-2.0) para capturar TODAS las detecciones candidatas y
    # despues filtrar por cada threshold sin tener que re-correr detectMultiScale
    all_scores_per_image = []
    for img_path in test_negatives:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if max(img.shape[:2]) > 640:
            s = 640 / max(img.shape[:2])
            img = cv2.resize(img, None, fx=s, fy=s)
        _, weights = hog.detectMultiScale(img, hitThreshold=-2.0, winStride=cfg["stride"], scale=1.2)
        all_scores_per_image.append([float(x) for x in weights] if len(weights) else [])

    results[name] = dict(pos_scores=pos_scores, neg_scores_per_image=all_scores_per_image)

print(f"\n{'='*70}")
print(f"{'umbral':<10}{'recall lateral':<18}{'%img con FP (lateral)'}")
for t in THRESHOLDS:
    recalls = {}
    for name in VIEWS:
        recalls[name] = (results[name]["pos_scores"] > t).mean() * 100

    n_img_with_fp = 0
    for i in range(len(test_negatives)):
        has_fp = False
        for name in VIEWS:
            scores_i = results[name]["neg_scores_per_image"][i]
            if any(s > t for s in scores_i):
                has_fp = True
                break
        if has_fp:
            n_img_with_fp += 1
    pct_fp = n_img_with_fp / len(test_negatives) * 100

    print(f"{t:<10.2f}{recalls['lateral']:<18.1f}{pct_fp:.1f}%")
