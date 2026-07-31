import gc
import json
import random
import shutil
import zipfile
from pathlib import Path
import cv2
import numpy as np

PROJECT_DIR = Path.cwd().parent
ARTIFACTS_DIR = PROJECT_DIR / "artifcats"
DATASET_DIR = Path.cwd() / "dataset"

# Ventanas HOG
WIN_SIZE_LATERAL = (128, 64)
WIN_SIZE_FRONTAL = (96, 96)

# Configurar directorios de salida
POS_RAW_LAT_DIR = DATASET_DIR / "positives_raw_lateral"
POS_RAW_FRONT_DIR = DATASET_DIR / "positives_raw_frontal"
NEG_RAW_LAT_DIR = DATASET_DIR / "negatives_raw_lateral"
NEG_RAW_FRONT_DIR = DATASET_DIR / "negatives_raw_frontal"

# 1. Eliminar carpetas aumentadas viejas para evitar confusiones
print("Limpiando directorios de aumento viejos...")
for d in (DATASET_DIR / "positives_aug_lateral", 
          DATASET_DIR / "positives_aug_frontal", 
          DATASET_DIR / "negatives_aug_lateral", 
          DATASET_DIR / "negatives_aug_frontal"):
    if d.exists():
        shutil.rmtree(d)
        print(f"  Eliminado: {d.name}")

# 2. Extracción de Señales de Pare/Alto (Stop signs) de COCO
def extract_coco_stop_signs():
    print("\n--- Extrayendo señales de ALTO (Stop Signs) desde COCO ---")
    val_dir = ARTIFACTS_DIR / "val2017"
    ann_zip = ARTIFACTS_DIR / "annotations_trainval2017.zip"
    if not (ann_zip.exists() and val_dir.exists()):
        print("  [Omitido] val2017 o zip de anotaciones no encontrado.")
        return 0
        
    with zipfile.ZipFile(ann_zip) as zf:
        with zf.open("annotations/instances_val2017.json") as f:
            coco = json.load(f)
            
    images_by_id = {im["id"]: im for im in coco["images"]}
    stop_sign_cat = next((c["id"] for c in coco["categories"] if "stop sign" in c["name"].lower()), None)
    
    if stop_sign_cat is None:
        print("  [Error] No se encontro la categoria 'stop sign' en el JSON.")
        return 0
        
    n_saved_lat = 0
    n_saved_front = 0
    
    # Filtrar anotaciones de señales de pare
    stop_anns = [a for a in coco["annotations"] if a["category_id"] == stop_sign_cat]
    random.shuffle(stop_anns)
    
    for ann in stop_anns:
        im_info = images_by_id.get(ann["image_id"])
        if im_info is None:
            continue
        x, y, w, h = ann["bbox"]
        if w <= 0 or h <= 0 or min(w, h) < 16:
            continue
            
        img_path = val_dir / im_info["file_name"]
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        # Recortar con leve padding
        ih, iw = img.shape[:2]
        pad_w, pad_h = w * 0.1, h * 0.1
        x0 = max(0, int(x - pad_w))
        y0 = max(0, int(y - pad_h))
        x1 = min(iw, int(x + w + pad_w))
        y1 = min(ih, int(y + h + pad_h))
        crop = img[y0:y1, x0:x1]
        
        if crop is None or crop.size == 0:
            continue
            
        # Guardar en ambas vistas
        crop_lat = cv2.resize(crop, WIN_SIZE_LATERAL)
        cv2.imwrite(str(NEG_RAW_LAT_DIR / f"coco_stopsign_{ann['id']}.jpg"), crop_lat)
        n_saved_lat += 1
        
        crop_front = cv2.resize(crop, WIN_SIZE_FRONTAL)
        cv2.imwrite(str(NEG_RAW_FRONT_DIR / f"coco_stopsign_{ann['id']}.jpg"), crop_front)
        n_saved_front += 1
        
    print(f"  Señales de ALTO extraídas: {n_saved_lat} lateral, {n_saved_front} frontal")
    return n_saved_lat

# 3. Extracción de Pavimento, Señales de Suelo y Fachadas de los Videos
def extract_from_videos():
    print("\n--- Extrayendo Pavimento y Fachadas desde los videos de trafico ---")
    video_files = [
        "trafico.webm",
        "CamionesFrontal.mp4",
        "carretera laterales.webm"
    ]
    
    n_pavement_lat = 0
    n_pavement_front = 0
    n_facade_lat = 0
    n_facade_front = 0
    
    for v_name in video_files:
        v_path = ARTIFACTS_DIR / "videos" / v_name
        if not v_path.exists():
            print(f"  [Omitido] Video no encontrado: {v_name}")
            continue
            
        print(f"  Procesando {v_name}...")
        cap = cv2.VideoCapture(str(v_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 1000
            
        # Muestrear frames uniformemente
        step = max(10, total_frames // 120)
        for i in range(120):
            frame_idx = i * step
            if frame_idx >= total_frames:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
                
            # Redimensionar a 640 de ancho para consistencia
            if frame.shape[1] > 640:
                double_s = 640.0 / frame.shape[1]
                frame = cv2.resize(frame, (640, int(frame.shape[0] * double_s)))
            
            ih, iw = frame.shape[:2]
            
            # A. Pavimento (30% inferior de la imagen - contiene asfalto y marcas viales)
            # Tomar parches del pavimento a diferentes posiciones de X
            pave_h = int(ih * 0.3)
            pave_y = int(ih * 0.7)
            for start_x in [0, int(iw * 0.3), int(iw * 0.6)]:
                pave_w = int(iw * 0.35)
                x1 = min(iw, start_x + pave_w)
                crop_pave = frame[pave_y:pave_y+pave_h, start_x:x1]
                if crop_pave.size > 0:
                    crop_pave_lat = cv2.resize(crop_pave, WIN_SIZE_LATERAL)
                    cv2.imwrite(str(NEG_RAW_LAT_DIR / f"vid_pave_{v_name.split('.')[0]}_{frame_idx}_{start_x}.jpg"), crop_pave_lat)
                    n_pavement_lat += 1
                    
                    crop_pave_front = cv2.resize(crop_pave, WIN_SIZE_FRONTAL)
                    cv2.imwrite(str(NEG_RAW_FRONT_DIR / f"vid_pave_{v_name.split('.')[0]}_{frame_idx}_{start_x}.jpg"), crop_pave_front)
                    n_pavement_front += 1
                    
            # B. Fachadas y Edificios (40% superior de la imagen - edificios, ventanas, cielo)
            fac_h = int(ih * 0.4)
            for start_x in [0, int(iw * 0.3), int(iw * 0.6)]:
                fac_w = int(iw * 0.35)
                x1 = min(iw, start_x + fac_w)
                crop_fac = frame[0:fac_h, start_x:x1]
                if crop_fac.size > 0:
                    crop_fac_lat = cv2.resize(crop_fac, WIN_SIZE_LATERAL)
                    cv2.imwrite(str(NEG_RAW_LAT_DIR / f"vid_facade_{v_name.split('.')[0]}_{frame_idx}_{start_x}.jpg"), crop_fac_lat)
                    n_facade_lat += 1
                    
                    crop_fac_front = cv2.resize(crop_fac, WIN_SIZE_FRONTAL)
                    cv2.imwrite(str(NEG_RAW_FRONT_DIR / f"vid_facade_{v_name.split('.')[0]}_{frame_idx}_{start_x}.jpg"), crop_fac_front)
                    n_facade_front += 1
                    
        cap.release()
        
    print(f"  Pavimento extraído: {n_pavement_lat} lateral, {n_pavement_front} frontal")
    print(f"  Edificios/Fachadas extraídos: {n_facade_lat} lateral, {n_facade_front} frontal")
    return n_pavement_lat + n_facade_lat

# 4. Extracción de Árboles, Bosques y Naturaleza desde Places365
def extract_places_nature():
    print("\n--- Extrayendo parches de ÁRBOLES y BOSQUES desde Places365 ---")
    places_dir = ARTIFACTS_DIR / "Places365"
    if not places_dir.exists():
        print("  [Omitido] Places365 no encontrado.")
        return 0
        
    # Palabras clave de naturaleza en Places365
    nature_keywords = ["forest", "mountain", "tree", "park", "field", "wood"]
    nature_paths = []
    for kw in nature_keywords:
        nature_paths += list(places_dir.rglob(f"*{kw}*/**/*.jpg"))
        nature_paths += list(places_dir.rglob(f"*{kw}*/*.jpg"))
        
    # Eliminar duplicados en la lista de paths
    nature_paths = list(set(nature_paths))
    
    if not nature_paths:
        print("  No se encontraron subdirectorios explicitos de naturaleza, buscando general...")
        # Si no hay rutas con palabras clave, tomar aleatorio de Places365
        nature_paths = list(places_dir.rglob("*.jpg"))
        
    random.shuffle(nature_paths)
    nature_sample = nature_paths[:800]
    
    n_saved_lat = 0
    n_saved_front = 0
    
    for img_path in nature_sample:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        ih, iw = img.shape[:2]
        
        # Recortar un parche central grande (bosque/árbol)
        h = int(ih * 0.7)
        w = int(iw * 0.7)
        x0 = (iw - w) // 2
        y0 = (ih - h) // 2
        crop = img[y0:y0+h, x0:x0+w]
        
        if crop is None or crop.size == 0:
            continue
            
        crop_lat = cv2.resize(crop, WIN_SIZE_LATERAL)
        cv2.imwrite(str(NEG_RAW_LAT_DIR / f"places_nature_{img_path.stem}.jpg"), crop_lat)
        n_saved_lat += 1
        
        crop_front = cv2.resize(crop, WIN_SIZE_FRONTAL)
        cv2.imwrite(str(NEG_RAW_FRONT_DIR / f"places_nature_{img_path.stem}.jpg"), crop_front)
        n_saved_front += 1
        
    print(f"  Árboles/Bosques extraídos: {n_saved_lat} lateral, {n_saved_front} frontal")
    return n_saved_lat

# Ejecutar las extracciones
extract_coco_stop_signs()
extract_from_videos()
extract_places_nature()

# Imprimir totales en negatives_raw
total_neg_lat = len(list(NEG_RAW_LAT_DIR.glob("*.jpg")))
total_neg_front = len(list(NEG_RAW_FRONT_DIR.glob("*.jpg")))
print(f"\n==========================================")
print(f" Dataset raw enriquecido con éxito!")
print(f"  Negativos raw lateral final: {total_neg_lat}")
print(f"  Negativos raw frontal final: {total_neg_front}")
print(f"==========================================")
