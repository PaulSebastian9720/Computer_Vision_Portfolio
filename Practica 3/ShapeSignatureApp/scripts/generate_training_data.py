"""
Genera training_data.h con los descriptores de Fourier de coordenadas complejas
extraídos del Flavia Dataset para las clases asignadas al Grupo G9.

Grupo G9:  Paul Sebastian Naspud Vivar | Jennyfer Camila Ramirez Saeteros
Clases:    24, 25, 26, 27, 28, 29, 30

Uso:
    cd Practica 3/ShapeSignatureApp/scripts
    pip install opencv-python numpy scikit-learn albumentations
    python generate_training_data.py
"""

import os
import sys
import csv
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

# ── Rutas absolutas ──────────────────────────────────────────────────────────
DATASET_DIR   = "/home/paul/universidad/septimo_v2/Vison_Computador/Practicas/Practica 3/Leaves"
CSV_PATH      = "/home/paul/universidad/septimo_v2/Vison_Computador/Practicas/Practica 3/Leaves/all.csv"
OUTPUT_HEADER = os.path.join(os.path.dirname(__file__),
                             "../app/src/main/cpp/training_data.h")

CLASSES       = [24, 25, 26, 27, 28, 29, 30]   # Clases G9
N_POINTS      = 256    # Puntos de remuestreo del contorno
N_DESC        = 12     # Armónicos del descriptor
RANDOM_STATE  = 42
TEST_SIZE     = 0.20   # 20 % para test

# ── Albumentations (opcional, mejora la generalización) ──────────────────────
try:
    import albumentations as A
    AUGMENT = True
    AUGMENT_COPIES = 4  # variantes por imagen original

    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=30, p=0.7, border_mode=cv2.BORDER_REFLECT),
        A.RandomBrightnessContrast(brightness_limit=0.2,
                                   contrast_limit=0.2, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
        A.Blur(blur_limit=3, p=0.3),
    ])
    print("[INFO] Albumentations disponible — se usará augmentación ×{}".format(AUGMENT_COPIES))
except ImportError:
    AUGMENT = False
    print("[WARN] Albumentations no instalado. Ejecuta: pip install albumentations")
    print("[INFO] Continuando sin augmentación...")

# ────────────────────────────────────────────────────────────────────────────

def resample_contour(contour, N):
    """Remuestrea el contorno a N puntos equiespaciados por longitud de arco."""
    pts = contour.reshape(-1, 2).astype(np.float32)
    # Contorno cerrado: añadir el primer punto al final para incluir el segmento de cierre
    pts_closed = np.vstack([pts, pts[0:1]])          # (M+1, 2)
    diffs      = np.diff(pts_closed, axis=0)         # (M, 2) — M segmentos
    seg_lens   = np.hypot(diffs[:, 0], diffs[:, 1])
    arc        = np.concatenate([[0], np.cumsum(seg_lens)])  # (M+1,)
    total      = arc[-1]
    if total < 1.0:
        return None
    targets = np.linspace(0, total, N, endpoint=False)
    xs = np.interp(targets, arc, pts_closed[:, 0])
    ys = np.interp(targets, arc, pts_closed[:, 1])
    return np.stack([xs, ys], axis=1)


def extract_descriptor(img_gray):
    """
    Extrae el descriptor de Fourier de coordenadas complejas de la imagen en gris.
    Pipeline idéntico al de native-lib.cpp:
      GaussianBlur → adaptiveThreshold → findContours → mayor contorno
      → centroide → señal compleja → DFT → normalizar por |F(1)| → magnitudes F(1..12)
    Retorna array de 12 floats o None si falla.
    """
    # 1. Reducción de ruido
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # 2. Umbral Otsu (fotos con fondo blanco uniforme del dataset Flavia)
    _, binary = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. Limpieza morfológica
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)

    # 4. Contornos externos
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    # Mayor contorno por área
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 200:
        return None

    # 5. Centroide
    M = cv2.moments(contour)
    if abs(M['m00']) < 1.0:
        return None
    xc = M['m10'] / M['m00']
    yc = M['m01'] / M['m00']

    # 6. Remuestrear a N_POINTS puntos
    pts = resample_contour(contour, N_POINTS)
    if pts is None:
        return None

    # 7. Señal compleja s(n) = (x(n)−xc) + j·(y(n)−yc)
    signal = (pts[:, 0] - xc) + 1j * (pts[:, 1] - yc)

    # 8. FFT
    F = np.fft.fft(signal)

    # 9. Normalizar por |F(1)|
    f1_mag = abs(F[1])
    if f1_mag < 1e-6:
        return None

    # 10. Descriptor: magnitudes de F(1)..F(12) normalizadas
    desc = np.abs(F[1: N_DESC + 1]) / f1_mag   # shape (12,)
    return desc.astype(np.float32)


def load_dataset():
    """Lee CSV, carga imágenes de las clases asignadas y extrae descriptores."""
    records = []
    with open(CSV_PATH, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = int(row['y'])
            if label in CLASSES:
                records.append((row['id'], label))

    print(f"[INFO] {len(records)} imágenes encontradas para clases {CLASSES}")

    X, y = [], []
    skipped = 0

    for filename, label in records:
        img_path = os.path.join(DATASET_DIR, filename)
        img = cv2.imread(img_path)
        if img is None:
            skipped += 1
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Imagen original
        desc = extract_descriptor(gray)
        if desc is not None:
            X.append(desc)
            y.append(label)

        # Augmentaciones con Albumentations
        if AUGMENT:
            for _ in range(AUGMENT_COPIES):
                augmented = transform(image=img)['image']
                gray_aug = cv2.cvtColor(augmented, cv2.COLOR_BGR2GRAY)
                desc_aug = extract_descriptor(gray_aug)
                if desc_aug is not None:
                    X.append(desc_aug)
                    y.append(label)

    print(f"[INFO] {len(X)} descriptores extraídos  ({skipped} imágenes saltadas)")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def write_header(X_train, y_train, X_test, y_test, path):
    """Escribe training_data.h con los datos de entrenamiento como arrays C++."""
    n_train = len(y_train)
    n_test  = len(y_test)

    lines = [
        "#pragma once",
        "// Autogenerado por scripts/generate_training_data.py",
        "// Grupo G9: Paul Sebastian Naspud Vivar | Jennyfer Camila Ramirez Saeteros",
        f"// Clases: {CLASSES}",
        f"// Train: {n_train} muestras  |  Test: {n_test} muestras",
        f"// Augmentacion: {'SI (Albumentations)' if AUGMENT else 'NO'}",
        "",
        f"static const int NUM_TRAIN      = {n_train};",
        f"static const int NUM_TEST       = {n_test};",
        f"static const int DESCRIPTOR_SIZE = {N_DESC};",
        "",
    ]

    # TRAIN_DESCRIPTORS
    lines.append(f"static const float TRAIN_DESCRIPTORS[{n_train}][{N_DESC}] = {{")
    for i, row in enumerate(X_train):
        vals = ", ".join(f"{v:.6f}f" for v in row)
        comma = "," if i < n_train - 1 else ""
        lines.append(f"    {{{vals}}}{comma}")
    lines.append("};")
    lines.append("")

    # TRAIN_LABELS
    labels_str = ", ".join(str(l) for l in y_train)
    lines.append(f"static const int TRAIN_LABELS[{n_train}] = {{{labels_str}}};")
    lines.append("")

    # TEST_DESCRIPTORS
    lines.append(f"static const float TEST_DESCRIPTORS[{n_test}][{N_DESC}] = {{")
    for i, row in enumerate(X_test):
        vals = ", ".join(f"{v:.6f}f" for v in row)
        comma = "," if i < n_test - 1 else ""
        lines.append(f"    {{{vals}}}{comma}")
    lines.append("};")
    lines.append("")

    # TEST_LABELS
    labels_str = ", ".join(str(l) for l in y_test)
    lines.append(f"static const int TEST_LABELS[{n_test}] = {{{labels_str}}};")
    lines.append("")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"[OK]   Escrito: {os.path.abspath(path)}")


def evaluate(X_train, y_train, X_test, y_test):
    """KNN-1 Euclidean sobre el conjunto de test para reportar accuracy."""
    correct = 0
    per_class = {c: [0, 0] for c in CLASSES}  # [correct, total]

    for i in range(len(y_test)):
        diffs  = X_train - X_test[i]
        dists  = np.sqrt((diffs ** 2).sum(axis=1))
        pred   = y_train[np.argmin(dists)]
        true   = y_test[i]
        per_class[true][1] += 1
        if pred == true:
            correct += 1
            per_class[true][0] += 1

    acc = correct / len(y_test) * 100
    print(f"\n[RESULTADO] Accuracy KNN-1 en test ({len(y_test)} muestras): {acc:.2f}%")
    print("  Por clase:")
    for c in CLASSES:
        ok, tot = per_class[c]
        if tot > 0:
            print(f"    Clase {c}: {ok}/{tot}  ({ok/tot*100:.1f}%)")
    return acc


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.path.isfile(CSV_PATH):
        print(f"[ERROR] No se encontró el CSV: {CSV_PATH}")
        sys.exit(1)

    print("[1/4] Cargando dataset...")
    X, y = load_dataset()

    if len(X) == 0:
        print("[ERROR] No se extrajeron descriptores. Revisa las rutas.")
        sys.exit(1)

    print("[2/4] Dividiendo en train/test (80/20 estratificado)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"       Train: {len(y_train)}  |  Test: {len(y_test)}")

    print("[3/4] Evaluando accuracy KNN-1...")
    evaluate(X_train, y_train, X_test, y_test)

    print("[4/4] Escribiendo training_data.h...")
    write_header(X_train, y_train, X_test, y_test, OUTPUT_HEADER)
    print("\n[DONE] Ahora ejecuta: ./gradlew assembleDebug")
