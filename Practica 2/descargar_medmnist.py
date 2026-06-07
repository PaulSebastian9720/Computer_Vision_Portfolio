"""
Descarga los 3 datasets MedMNIST necesarios para la Práctica 2.
Total: ~114 MB  |  Sin registro  |  Fuente: Zenodo (Yang et al., 2023)
Uso:  python descargar_medmnist.py
"""
import urllib.request, os, time

DEST_DIR = "medmnist_data"
BASE_URL = "https://zenodo.org/record/10519652/files"
HEADERS  = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0"}

DATASETS = [
    ("pneumoniamnist_64.npz", "Radiografías de tórax  (PneumoniaMNIST, 20.6 MB)"),
    ("organcmnist_64.npz",    "Tomografías abdominales (OrganCMNIST,    80.3 MB)"),
    ("retinamnist_64.npz",    "Angiografías de retina  (RetinaMNIST,    13.2 MB)"),
]

os.makedirs(DEST_DIR, exist_ok=True)

for fname, desc in DATASETS:
    dest = os.path.join(DEST_DIR, fname)
    if os.path.exists(dest):
        print(f"  ✓  {desc}  (ya descargado)")
        continue

    url = f"{BASE_URL}/{fname}?download=1"
    print(f"  ↓  {desc}")
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=180) as r, open(dest, "wb") as f:
            total = int(r.headers.get("Content-Length", 0))
            downloaded = 0
            while chunk := r.read(524288):  # 512 KB chunks
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r     {pct:.0f}%  ({downloaded//1048576} MB)", end="", flush=True)
        print(f"\r     Listo: {os.path.getsize(dest)//1048576} MB guardados en {dest}")
    except Exception as e:
        print(f"\n     Error: {e}")
    time.sleep(1)

print("\nDescarga completada. Ahora puedes ejecutar el notebook.")
