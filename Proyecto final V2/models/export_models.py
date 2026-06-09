#!/usr/bin/env python3
"""
Extrae los modelos TFLite del paquete mediapipe instalado y los convierte a ONNX.
No necesita mediapipe en runtime C++. Solo se corre una vez.
Requiere: pip install mediapipe tf2onnx onnx
"""
import sys, os, subprocess, pathlib, urllib.request, shutil

SCRIPT_DIR = pathlib.Path(__file__).parent
OUT_PALM = SCRIPT_DIR / "palm_detection.onnx"
OUT_LM   = SCRIPT_DIR / "hand_landmark.onnx"

# URLs de fallback (modelos TFLite oficiales de Google/MediaPipe en GitHub)
PALM_URL = "https://github.com/google/mediapipe/raw/master/mediapipe/modules/palm_detection/palm_detection_lite.tflite"
LM_URL   = "https://github.com/google/mediapipe/raw/master/mediapipe/modules/hand_landmark/hand_landmark_lite.tflite"


def ensure_deps():
    for pkg in ("tf2onnx", "onnx"):
        try:
            __import__(pkg)
        except ImportError:
            print(f"Instalando {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])


def find_in_mediapipe():
    """Busca .tflite en el paquete mediapipe instalado."""
    try:
        import mediapipe as mp
    except ImportError:
        return None, None

    base = pathlib.Path(mp.__file__).parent
    palm = lm = None

    candidates_palm = list(base.rglob("palm_detection_lite.tflite"))
    candidates_lm   = list(base.rglob("hand_landmark_lite.tflite"))

    # Fallback nombres alternativos
    if not candidates_palm:
        candidates_palm = list(base.rglob("palm_detection*.tflite"))
    if not candidates_lm:
        candidates_lm = (list(base.rglob("hand_landmark*.tflite")) +
                         list(base.rglob("hand_lm*.tflite")))

    if candidates_palm:
        palm = candidates_palm[0]
        print(f"  palm TFLite: {palm}")
    if candidates_lm:
        lm = candidates_lm[0]
        print(f"  landmark TFLite: {lm}")

    return palm, lm


def download_tflite(url, dest: pathlib.Path):
    print(f"  Descargando {dest.name} desde GitHub...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f)
        size = dest.stat().st_size
        if size < 1000:
            print(f"  Archivo muy pequeño ({size}B), puede ser un error.")
            return False
        print(f"  OK ({size//1024} KB)")
        return True
    except Exception as e:
        print(f"  Error al descargar: {e}")
        return False


def convert(tflite: pathlib.Path, onnx: pathlib.Path):
    print(f"Convirtiendo {tflite.name} → {onnx.name} ...")
    r = subprocess.run(
        [sys.executable, "-m", "tf2onnx.convert",
         "--tflite", str(tflite),
         "--output", str(onnx),
         "--opset", "11"],
        capture_output=True, text=True
    )
    if r.returncode == 0:
        size = onnx.stat().st_size
        print(f"  OK ({size//1024} KB)")
        return True
    else:
        print(f"  FALLO:\n{r.stderr[-500:]}")
        return False


def main():
    print("=== Exportando modelos MediaPipe → ONNX ===\n")

    print("1. Verificando dependencias...")
    ensure_deps()
    print("   OK\n")

    tmp_dir = SCRIPT_DIR / "_tflite_tmp"
    tmp_dir.mkdir(exist_ok=True)

    # ── Buscar en mediapipe instalado ──────────────────────────────────────────
    print("2. Buscando .tflite en paquete mediapipe...")
    palm_tflite, lm_tflite = find_in_mediapipe()

    # ── Fallback: descargar desde GitHub si no se encuentran ──────────────────
    if not palm_tflite:
        print("  No encontrado localmente. Intentando descargar...")
        palm_tflite = tmp_dir / "palm_detection_lite.tflite"
        if not download_tflite(PALM_URL, palm_tflite):
            palm_tflite = None

    if not lm_tflite:
        print("  No encontrado localmente. Intentando descargar...")
        lm_tflite = tmp_dir / "hand_landmark_lite.tflite"
        if not download_tflite(LM_URL, lm_tflite):
            lm_tflite = None

    if not palm_tflite or not lm_tflite:
        print("\nERROR: No se pudieron obtener los modelos TFLite.")
        print("Instala mediapipe: pip install mediapipe")
        sys.exit(1)

    print()

    # ── Convertir ─────────────────────────────────────────────────────────────
    print("3. Convirtiendo a ONNX...")
    ok1 = convert(palm_tflite, OUT_PALM)
    ok2 = convert(lm_tflite,   OUT_LM)

    # Limpieza
    shutil.rmtree(tmp_dir, ignore_errors=True)

    if ok1 and ok2:
        print("\n=== Listo ===")
        print(f"  {OUT_PALM}")
        print(f"  {OUT_LM}")
        print("\nEjecuta ahora: make run")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
