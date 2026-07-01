"""
Genera visualizaciones del pipeline de procesamiento para el informe.
Para cada clase 24-30 produce:
  (a) Imagen original
  (b) Suavizada + detección de bordes (Canny)
  (c) Contorno detectado y centroide
  (d) Firma de la forma  s(n) = x(n)-xc + j*(y(n)-yc)
Y un panel 2x2 combinado (listo para el informe PDF).
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import csv
import os
import sys

# ── Rutas ─────────────────────────────────────────────────────────────────────
DATASET_DIR = "/home/paul/universidad/septimo_v2/Vison_Computador/Practicas/Practica 3/Leaves"
CSV_PATH    = "/home/paul/universidad/septimo_v2/Vison_Computador/Practicas/Practica 3/Leaves/all.csv"
OUT_DIR     = "/home/paul/universidad/septimo_v2/Vison_Computador/Practicas/Practica 3/informe_visualizaciones"
CLASES      = [24, 25, 26, 27, 28, 29, 30]
N_POINTS    = 256   # puntos de remuestreo del contorno

# Nombres de especies Flavia Dataset (Wu et al., 2007)
SPECIES = {
    24: "Pittosporum tobira",
    25: "Nelumbo nucifera",
    26: "Acer palmatum",
    27: "Diospyros kaki",
    28: "Populus tomentosa",
    29: "Armeniaca mume",
    30: "Cinnamomum japonicum",
}

os.makedirs(OUT_DIR, exist_ok=True)
for c in CLASES:
    os.makedirs(os.path.join(OUT_DIR, f"clase_{c:02d}"), exist_ok=True)

# ── Leer CSV y agrupar por clase ───────────────────────────────────────────────
class_files = {c: [] for c in CLASES}
with open(CSV_PATH) as f:
    for row in csv.DictReader(f):
        c = int(row["y"])
        if c in CLASES:
            path = os.path.join(DATASET_DIR, row["id"])
            if os.path.exists(path):
                class_files[c].append(path)

for c in CLASES:
    class_files[c].sort()
    print(f"Clase {c} ({SPECIES[c]}): {len(class_files[c])} imágenes")

# ── Helpers ────────────────────────────────────────────────────────────────────
def resample_contour(pts, n=N_POINTS):
    pts = pts.reshape(-1, 2).astype(np.float32)
    closed = np.vstack([pts, pts[0:1]])
    diffs  = np.diff(closed, axis=0)
    seg_len = np.sqrt((diffs**2).sum(axis=1))
    arc = np.concatenate([[0], np.cumsum(seg_len)])
    total = arc[-1]
    if total < 1e-6:
        return pts
    t_uniform = np.linspace(0, total, n, endpoint=False)
    rx = np.interp(t_uniform, arc, closed[:, 0])
    ry = np.interp(t_uniform, arc, closed[:, 1])
    return np.stack([rx, ry], axis=1)


def pipeline(img_path):
    """Devuelve todos los intermedios del pipeline."""
    img_bgr  = cv2.imread(img_path)
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur     = cv2.GaussianBlur(gray, (5, 5), 0)
    canny    = cv2.Canny(blur, 30, 100)

    _, binary = cv2.threshold(blur, 0, 255,
                              cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    xc = M["m10"] / M["m00"]
    yc = M["m01"] / M["m00"]

    pts     = resample_contour(contour.reshape(-1, 2))
    signal  = (pts[:, 0] - xc) + 1j * (pts[:, 1] - yc)

    return dict(
        img_rgb=img_rgb,
        blur=blur,
        canny=canny,
        binary=binary,
        contour=contour,
        xc=xc, yc=yc,
        pts=pts,
        signal=signal,
    )


def make_panel(data, clase, species, filename, img_path):
    """Genera el panel 2x2 estilo guía."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9),
                             facecolor="#111111")
    fig.suptitle(
        f"Clase {clase} — {species}",
        color="white", fontsize=14, fontweight="bold", y=0.98
    )

    ax = axes[0, 0]
    ax.imshow(data["img_rgb"])
    ax.set_title("(a) Imagen original", color="white", fontsize=11)
    ax.axis("off")

    ax = axes[0, 1]
    canny_color = np.zeros((*data["canny"].shape, 3), dtype=np.uint8)
    canny_color[data["canny"] > 0] = [255, 255, 255]
    ax.imshow(canny_color, cmap="gray")
    ax.set_title("(b) Suavizada y detección de bordes (Canny)", color="white", fontsize=11)
    ax.axis("off")

    ax = axes[1, 0]
    h, w = data["img_rgb"].shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.drawContours(canvas, [data["contour"]], -1, (255, 50, 50), 2)
    cv2.circle(canvas, (int(data["xc"]), int(data["yc"])), 8, (255, 200, 0), -1)
    cv2.drawMarker(canvas, (int(data["xc"]), int(data["yc"])),
                   (255, 200, 0), cv2.MARKER_CROSS, 20, 2)
    ax.imshow(canvas)
    ax.set_title("(c) Centroide y contorno detectado", color="white", fontsize=11)
    ax.axis("off")

    ax = axes[1, 1]
    ax.set_facecolor("black")
    n = len(data["signal"])
    idx = np.arange(n)
    ax.plot(idx, data["signal"].real, color="#FF4444", linewidth=1.2,
            label="Re s(n) = x(n)−xc")
    ax.plot(idx, data["signal"].imag, color="#44AAFF", linewidth=1.2,
            linestyle="--", label="Im s(n) = y(n)−yc")
    ax.set_title("(d) Firma de la forma de la hoja", color="white", fontsize=11)
    ax.set_xlabel("n", color="gray")
    ax.set_ylabel("s(n)", color="gray")
    ax.tick_params(colors="gray")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")
    ax.legend(facecolor="#222222", labelcolor="white", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = os.path.join(OUT_DIR, f"clase_{clase:02d}", filename)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#111111")
    plt.close(fig)
    return out_path


def save_individual(data, clase, base):
    """Guarda las 4 imágenes por separado (útil para insertar en LaTeX/Word)."""
    subdir = os.path.join(OUT_DIR, f"clase_{clase:02d}")

    # (a) original
    cv2.imwrite(os.path.join(subdir, f"{base}_a_original.jpg"),
                cv2.cvtColor(data["img_rgb"], cv2.COLOR_RGB2BGR))

    # (b) Canny
    cv2.imwrite(os.path.join(subdir, f"{base}_b_canny.jpg"), data["canny"])

    # (c) contorno + centroide
    h, w = data["img_rgb"].shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.drawContours(canvas, [data["contour"]], -1, (0, 50, 255), 2)
    cv2.drawMarker(canvas, (int(data["xc"]), int(data["yc"])),
                   (0, 200, 255), cv2.MARKER_CROSS, 20, 2)
    cv2.imwrite(os.path.join(subdir, f"{base}_c_contorno.jpg"), canvas)

    # (d) firma → guardada como PNG vía matplotlib
    fig, ax = plt.subplots(figsize=(7, 3), facecolor="black")
    ax.set_facecolor("black")
    n   = len(data["signal"])
    idx = np.arange(n)
    ax.plot(idx, data["signal"].real, color="#FF4444", linewidth=1.2)
    ax.plot(idx, data["signal"].imag, color="#4488FF", linewidth=1.2, linestyle="--")
    ax.tick_params(colors="gray")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    plt.tight_layout()
    plt.savefig(os.path.join(subdir, f"{base}_d_firma.png"),
                dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)


# ── Procesar ──────────────────────────────────────────────────────────────────
# Por defecto: 3 muestras representativas por clase
N_SAMPLES = 3

for clase in CLASES:
    files = class_files[clase]
    if not files:
        print(f"  Clase {clase}: sin imágenes, saltando")
        continue

    # Tomar N_SAMPLES distribuidas uniformemente
    indices = np.linspace(0, len(files) - 1, N_SAMPLES, dtype=int)
    samples = [files[i] for i in indices]

    print(f"\nClase {clase} ({SPECIES[clase]}):")
    for i, img_path in enumerate(samples):
        fname = os.path.basename(img_path)
        base  = fname.replace(".jpg", "")
        data  = pipeline(img_path)
        if data is None:
            print(f"  [{i+1}] {fname} — sin contorno, saltando")
            continue

        panel_path = make_panel(data, clase, SPECIES[clase],
                                f"{base}_panel.png", img_path)
        save_individual(data, clase, base)
        print(f"  [{i+1}] {fname} → panel y 4 imágenes guardadas")

print(f"\nListo. Visualizaciones en:\n  {OUT_DIR}")
