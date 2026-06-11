import cv2
import numpy as np
import os
import glob
import threading
import time

# ==========================================
# 1. CONFIGURACIÓN DEL PROYECTO
# ==========================================
URL_IZQ = "http://192.168.0.120/stream"  
URL_DER = "http://192.168.0.121/stream"  

FILAS = 6          
COLUMNAS = 9       
TAMANO_CUADRADO = 26.0

os.makedirs("capturas/izquierda", exist_ok=True)
os.makedirs("capturas/derecha", exist_ok=True)

class StreamReceiver(threading.Thread):
    def __init__(self, url):
        super().__init__()
        self.url = url
        self.frame = None
        self.running = True
        self.daemon = True
        self.lock = threading.Lock()

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

print("\nLevantando hilos de captura optimizados...")
hilo_izq = StreamReceiver(URL_IZQ)
hilo_der = StreamReceiver(URL_DER)

hilo_izq.start()
hilo_der.start()

time.sleep(2)

contador = 0
print("\n=== FASE 1: CAPTURA CON CONFIRMACIÓN VISUAL ===")
print("-> Presiona 's' para validar. Si es correcto, verás las esquinas dibujadas por 1.5 segundos.")
print("-> Presiona 'q' para iniciar el procesamiento matemático final.\n")

while True:
    frame_i = hilo_izq.get_frame()
    frame_d = hilo_der.get_frame()
    
    if frame_i is None or frame_d is None:
        time.sleep(0.01)
        continue

    # El video en vivo sigue limpio y rápido
    cv2.imshow("Camara Izquierda", frame_i)
    cv2.imshow("Camara Derecha", frame_d)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('s'):
        print(f"\n[INFO] Analizando par {contador}...")
        
        gray_i = cv2.cvtColor(frame_i, cv2.COLOR_BGR2GRAY)
        gray_d = cv2.cvtColor(frame_d, cv2.COLOR_BGR2GRAY)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        gray_i_enhanced = clahe.apply(gray_i)
        gray_d_enhanced = clahe.apply(gray_d)
        
        flags_deteccion = (
            cv2.CALIB_CB_ADAPTIVE_THRESH + 
            cv2.CALIB_CB_NORMALIZE_IMAGE + 
            cv2.CALIB_CB_FAST_CHECK
        )
        
        ret_i, corners_i = cv2.findChessboardCorners(gray_i_enhanced, (COLUMNAS, FILAS), None, flags=flags_deteccion)
        ret_d, corners_d = cv2.findChessboardCorners(gray_d_enhanced, (COLUMNAS, FILAS), None, flags=flags_deteccion)
        
        if ret_i and ret_d:
            # Creamos copias para dibujar las esquinas encima sin alterar la foto original que se guarda
            preview_i = frame_i.copy()
            preview_d = frame_d.copy()
            
            cv2.drawChessboardCorners(preview_i, (COLUMNAS, FILAS), corners_i, ret_i)
            cv2.drawChessboardCorners(preview_d, (COLUMNAS, FILAS), corners_d, ret_d)
            
            # Mostramos el resultado coloreado en la pantalla para tu tranquilidad
            cv2.imshow("Camara Izquierda", preview_i)
            cv2.imshow("Camara Derecha", preview_d)
            
            # Guardamos las imágenes originales limpias
            cv2.imwrite(f"capturas/izquierda/frame_{contador}.png", frame_i)
            cv2.imwrite(f"capturas/derecha/frame_{contador}.png", frame_d)
            
            print(f"[OK] ¡Par {contador} CAPTURADO CON ÉXITO! (Mostrando feedback...)")
            contador += 1
            
            # Pausa el video por 1.5 segundos para que alcances a ver los rectángulos de colores
            cv2.waitKey(1500)
        else:
            print("[ALERTA] Falló la detección.")
            print(f" -> Izquierda: {'OK' if ret_i else 'FALLÓ'} | Derecha: {'OK' if ret_d else 'FALLÓ'}")
            
    elif key == ord('q'):
        break

hilo_izq.stop()
hilo_der.stop()
cv2.destroyAllWindows()

if contador < 10:
    print(f"\n[ERROR] Solo guardaste {contador} pares. Necesitas mínimo 10.")
    exit()

# ==========================================
# 2. FASE DE CÓMPUTO MATEMÁTICO (Calibración)
# ==========================================
print("\n=== FASE 2: CALIBRACIÓN MATEMÁTICA ESTÉREO ===")
print("Procesando matrices ópticas...")

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((FILAS * COLUMNAS, 3), np.float32)
objp[:, :2] = np.mgrid[0:COLUMNAS, 0:FILAS].T.reshape(-1, 2) * TAMANO_CUADRADO

objpoints = []     
imgpoints_l = []   
imgpoints_r = []   

images_l = sorted(glob.glob('capturas/izquierda/*.png'))
images_r = sorted(glob.glob('capturas/derecha/*.png'))

for img_l_path, img_r_path in zip(images_l, images_r):
    img_l = cv2.imread(img_l_path)
    img_r = cv2.imread(img_r_path)
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray_l_enh = clahe.apply(gray_l)
    gray_r_enh = clahe.apply(gray_r)

    ret_l, corners_l = cv2.findChessboardCorners(gray_l_enh, (COLUMNAS, FILAS), None, flags=flags_deteccion)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r_enh, (COLUMNAS, FILAS), None, flags=flags_deteccion)

    if ret_l and ret_r:
        objpoints.append(objp)
        corners_l2 = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
        corners_r2 = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)
        imgpoints_l.append(corners_l2)
        imgpoints_r.append(corners_r2)

print("-> Calculando parámetros intrínsecos (K)...")
shape = gray_l.shape[::-1]
ret, K_l, dist_l, _, _ = cv2.calibrateCamera(objpoints, imgpoints_l, shape, None, None)
ret, K_r, dist_r, _, _ = cv2.calibrateCamera(objpoints, imgpoints_r, shape, None, None)

print("-> Ejecutando stereoCalibrate (R y T)...")
flags_calib = cv2.CALIB_FIX_INTRINSIC  
stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

ret, K_l, dist_l, K_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_l, imgpoints_r,
    K_l, dist_l, K_r, dist_r,
    shape, criteria=stereocalib_criteria, flags=flags_calib
)

print("-> Guardando en parametros_estereo.yml...")
fs = cv2.FileStorage("stereo_calib.yml", cv2.FILE_STORAGE_WRITE)
fs.write("K_l", K_l)
fs.write("dist_l", dist_l)
fs.write("K_r", K_r)
fs.write("dist_r", dist_r)
fs.write("R", R)
fs.write("T", T)
fs.release()

print("\n[ÉXITO TOTAL] Archivo YML generado.")
