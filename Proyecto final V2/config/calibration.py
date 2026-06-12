import cv2
import numpy as np
import os
import glob
import threading
import time

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================
URL_IZQ = "http://10.194.62.47/stream"
URL_DER = "http://10.194.62.63/stream"   

FILAS           = 6
COLUMNAS        = 9
TAMANO_CUADRADO = 25.0   # mm
MIN_PARES       = 15 

# Rutas relativas al directorio de este script (funciona desde cualquier CWD)
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DIR_IZQ      = os.path.join(SCRIPT_DIR, "capturas", "izquierda")
DIR_DER      = os.path.join(SCRIPT_DIR, "capturas", "derecha")
YAML_SALIDA  = os.path.join(SCRIPT_DIR, "stereo_calib.yml")

os.makedirs(DIR_IZQ, exist_ok=True)
os.makedirs(DIR_DER, exist_ok=True)

# Flags de detección de esquinas — definidos una sola vez
FLAGS_DETECCION = (
    cv2.CALIB_CB_ADAPTIVE_THRESH +
    cv2.CALIB_CB_NORMALIZE_IMAGE +
    cv2.CALIB_CB_FAST_CHECK
)

# Criterio de refinamiento subpíxel
CRITERIA_SUBPIX = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# ==========================================
# 2. HILO DE CAPTURA DE STREAM
# ==========================================
class StreamReceiver(threading.Thread):
    def __init__(self, url):
        super().__init__()
        self.url     = url
        self.frame   = None
        self.running = True
        self.daemon  = True
        self.lock    = threading.Lock()

    def run(self):
        cap = cv2.VideoCapture(self.url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        while self.running:
            ret, img = cap.read()
            if ret and img is not None:
                with self.lock:
                    self.frame = img
            else:
                time.sleep(0.5)
                cap.open(self.url)
        cap.release()

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False

# ==========================================
# 3. FASE 1 — CAPTURA DE PARES
# ==========================================
print("\nConectando a cámaras...")
hilo_izq = StreamReceiver(URL_IZQ)
hilo_der = StreamReceiver(URL_DER)
hilo_izq.start()
hilo_der.start()
time.sleep(2)

contador = 0
print("\n=== FASE 1: CAPTURA ===")
print("  's' → capturar par  |  'q' → terminar y calibrar\n")

while True:
    frame_i = hilo_izq.get_frame()
    frame_d = hilo_der.get_frame()

    if frame_i is None or frame_d is None:
        time.sleep(0.01)
        continue

    cv2.imshow("Izquierda", frame_i)
    cv2.imshow("Derecha",   frame_d)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        print(f"[INFO] Analizando par {contador}...")
        gray_i = cv2.cvtColor(frame_i, cv2.COLOR_BGR2GRAY)
        gray_d = cv2.cvtColor(frame_d, cv2.COLOR_BGR2GRAY)

        clahe          = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray_i_enh     = clahe.apply(gray_i)
        gray_d_enh     = clahe.apply(gray_d)

        ret_i, corners_i = cv2.findChessboardCorners(gray_i_enh, (COLUMNAS, FILAS), None, flags=FLAGS_DETECCION)
        ret_d, corners_d = cv2.findChessboardCorners(gray_d_enh, (COLUMNAS, FILAS), None, flags=FLAGS_DETECCION)

        if ret_i and ret_d:
            preview_i = frame_i.copy()
            preview_d = frame_d.copy()
            cv2.drawChessboardCorners(preview_i, (COLUMNAS, FILAS), corners_i, ret_i)
            cv2.drawChessboardCorners(preview_d, (COLUMNAS, FILAS), corners_d, ret_d)
            cv2.imshow("Izquierda", preview_i)
            cv2.imshow("Derecha",   preview_d)

            cv2.imwrite(os.path.join(DIR_IZQ, f"frame_{contador}.png"), frame_i)
            cv2.imwrite(os.path.join(DIR_DER, f"frame_{contador}.png"), frame_d)
            print(f"[OK] Par {contador} guardado")
            contador += 1
            cv2.waitKey(1500)
        else:
            print(f"[FALLO] Izq: {'OK' if ret_i else 'no'} | Der: {'OK' if ret_d else 'no'}")

    elif key == ord('q'):
        break

hilo_izq.stop()
hilo_der.stop()
cv2.destroyAllWindows()

if contador < MIN_PARES:
    print(f"\n[ERROR] Solo {contador} pares. Necesitas mínimo {MIN_PARES}.")
    exit(1)

# ==========================================
# 4. FASE 2 — CALIBRACIÓN MATEMÁTICA OPTIMIZADA
# ==========================================
print(f"\n=== FASE 2: CALIBRACIÓN ({contador} pares) ===")

objp = np.zeros((FILAS * COLUMNAS, 3), np.float32)
objp[:, :2] = np.mgrid[0:COLUMNAS, 0:FILAS].T.reshape(-1, 2) * TAMANO_CUADRADO

objpoints_raw    = []
imgpoints_l_raw  = []
imgpoints_r_raw  = []

images_l = sorted(glob.glob(os.path.join(DIR_IZQ, "*.png")))
images_r = sorted(glob.glob(os.path.join(DIR_DER, "*.png")))

img_shape = None
for path_l, path_r in zip(images_l, images_r):
    img_l  = cv2.imread(path_l)
    img_r  = cv2.imread(path_r)
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    img_shape = gray_l.shape[::-1]

    clahe      = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_l_enh = clahe.apply(gray_l)
    gray_r_enh = clahe.apply(gray_r)

    ret_l, corners_l = cv2.findChessboardCorners(gray_l_enh, (COLUMNAS, FILAS), None, flags=FLAGS_DETECCION)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r_enh, (COLUMNAS, FILAS), None, flags=FLAGS_DETECCION)

    if ret_l and ret_r:
        # Refinar subpíxel sobre la imagen original en escala de grises
        corners_l2 = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), CRITERIA_SUBPIX)
        corners_r2 = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), CRITERIA_SUBPIX)
        objpoints_raw.append(objp)
        imgpoints_l_raw.append(corners_l2)
        imgpoints_r_raw.append(corners_r2)

# ── PASO 1: Calibración preliminar para filtrado estricto ──
# Evitamos distorsiones de alto orden (K3, K4, K5) para mayor estabilidad en lentes comunes
flags_indiv = cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5
criteria_calib = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

print("  Ejecutando calibración preliminar para filtrado...")
_, K_l_pre, dist_l_pre, _, _ = cv2.calibrateCamera(objpoints_raw, imgpoints_l_raw, img_shape, None, None, flags=flags_indiv, criteria=criteria_calib)
_, K_r_pre, dist_r_pre, _, _ = cv2.calibrateCamera(objpoints_raw, imgpoints_r_raw, img_shape, None, None, flags=flags_indiv, criteria=criteria_calib)

# Filtrar con umbral estricto (< 0.8 px) antes de computar las intrínsecas finales
pares_buenos_obj = []
pares_buenos_l   = []
pares_buenos_r   = []

print("  Filtrando pares con error de reproyección alto...")
for i, (op, pl, pr) in enumerate(zip(objpoints_raw, imgpoints_l_raw, imgpoints_r_raw)):
    _, rvec_l, tvec_l = cv2.solvePnP(op, pl, K_l_pre, dist_l_pre)
    proj_l, _         = cv2.projectPoints(op, rvec_l, tvec_l, K_l_pre, dist_l_pre)
    err_l             = np.mean(np.linalg.norm(pl.reshape(-1, 2) - proj_l.reshape(-1, 2), axis=1))

    _, rvec_r, tvec_r = cv2.solvePnP(op, pr, K_r_pre, dist_r_pre)
    proj_r, _         = cv2.projectPoints(op, rvec_r, tvec_r, K_r_pre, dist_r_pre)
    err_r             = np.mean(np.linalg.norm(pr.reshape(-1, 2) - proj_r.reshape(-1, 2), axis=1))

    err_medio = (err_l + err_r) / 2
    if err_medio < 0.8:  
        pares_buenos_obj.append(op)
        pares_buenos_l.append(pl)
        pares_buenos_r.append(pr)
    else:
        print(f"    Par {i} descartado por error alto ({err_medio:.2f} px)")

print(f"  Pares limpios conservados: {len(pares_buenos_obj)} / {len(objpoints_raw)}")

if len(pares_buenos_obj) < 8:
    print("[ERROR] Muy pocos pares limpios pasaron el filtro. Recaptura con mejor estabilidad e iluminación.")
    exit(1)

# ── PASO 2: Calibración individual DEFINITIVA solo con pares limpios ──
print("  Recalculando intrínsecas definitivas...")
rmse_l, K_l, dist_l, _, _ = cv2.calibrateCamera(pares_buenos_obj, pares_buenos_l, img_shape, None, None, flags=flags_indiv, criteria=criteria_calib)
rmse_r, K_r, dist_r, _, _ = cv2.calibrateCamera(pares_buenos_obj, pares_buenos_r, img_shape, None, None, flags=flags_indiv, criteria=criteria_calib)
print(f"  RMSE limpio → izq: {rmse_l:.4f} px | der: {rmse_r:.4f} px")

# ── PASO 3: Calibración estéreo refinando ligeramente las intrínsecas ──
print("  Ejecutando stereoCalibrate con intrínsecas como guess inicial...")
# CALIB_USE_INTRINSIC_GUESS: usa las K calculadas arriba como punto de partida
# y las refina levemente junto con R y T. Evita que errores en K se absorban en T
# (bug de FIX_INTRINSIC: baseline resultaba ~40mm en vez de 80mm).
flags_stereo = cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5
criteria_stereo = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 300, 1e-6)

rmse_stereo, K_l, dist_l, K_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
    pares_buenos_obj, pares_buenos_l, pares_buenos_r,
    K_l, dist_l, K_r, dist_r,
    img_shape,
    criteria=criteria_stereo,
    flags=flags_stereo
)
print(f"  RMSE estéreo final: {rmse_stereo:.4f} px")

if rmse_stereo > 0.5:
    print("  [AVISO] RMSE > 0.5px — calibración aceptable pero mejorable.")
elif rmse_stereo < 0.3:
    print("  [EXCELENTE] RMSE < 0.3px")
else:
    print("  [BUENO] RMSE en rango óptimo.")

# ==========================================
# 5. GUARDAR YAML
# ==========================================
fs = cv2.FileStorage(YAML_SALIDA, cv2.FILE_STORAGE_WRITE)
fs.write("K_l",    K_l)
fs.write("dist_l", dist_l)
fs.write("K_r",    K_r)
fs.write("dist_r", dist_r)
fs.write("R",      R)
fs.write("T",      T)
fs.release()
print(f"\n  Guardado exitoso en: {YAML_SALIDA}")

# ==========================================
# 6. RESUMEN DE CALIDAD
# ==========================================
T_flat   = T.flatten()
baseline = np.linalg.norm(T_flat)
angulo   = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
BASELINE_FISICO_MM = 81.0 
err_base = (baseline - BASELINE_FISICO_MM) / BASELINE_FISICO_MM * 100
diff_foc = abs(K_l[0, 0] - K_r[0, 0]) / K_l[0, 0] * 100

print("\n╔══════════════════════════════════════════════╗")
print("║  RESUMEN DE CALIDAD                          ║")
print("╠══════════════════════════════════════════════╣")
print(f"║  RMSE estéreo:   {rmse_stereo:6.4f} px                  ║")
print(f"║  Baseline:       {baseline:6.2f} mm  (error {err_base:+.1f}%)      ║")
print(f"║  T vertical:     {T_flat[1]:+6.2f} mm                   ║")
print(f"║  T profundidad:  {T_flat[2]:+6.2f} mm                   ║")
print(f"║  Ángulo R:       {angulo:6.1f} °                      ║")
print(f"║  Diff focales:   {diff_foc:6.1f} %                      ║")
print("╠══════════════════════════════════════════════╣")

# Veredicto automático con umbrales ajustados
problemas = []
if rmse_stereo > 0.6:   problemas.append(f"RMSE alto ({rmse_stereo:.2f}px) — recaptura pares")
if abs(err_base) > 15:  problemas.append(f"Baseline muy errado ({err_base:+.1f}%) — distancia física errada")
if abs(T_flat[1]) > 8:  problemas.append(f"Desplazamiento vertical alto ({T_flat[1]:.1f}mm)")
if abs(T_flat[2]) > 15: problemas.append(f"T[2] de profundidad alto ({T_flat[2]:.1f}mm) — no coplanares")
if angulo > 5:          problemas.append(f"Ángulo grande ({angulo:.1f}°) — ajusta soporte físico")
if diff_foc > 5:        problemas.append(f"Focales distintas ({diff_foc:.1f}%) — bloquea enfoque/zoom")

if not problemas:
    print("║  ESTADO: BUENA CALIBRACIÓN                  ║")
else:
    print("║  ESTADO: REVISAR LOS SIGUIENTES PUNTOS:     ║")
    for p in problemas:
        linea = f"║    • {p}"
        print(f"{linea:<46}║")

print("╚══════════════════════════════════════════════╝")
