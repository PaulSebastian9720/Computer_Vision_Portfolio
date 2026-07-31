import json
import random
import shutil
import zipfile
import tarfile
from pathlib import Path
import cv2
import numpy as np

PROJECT_DIR = Path.cwd().parent
ARTIFACTS_DIR = PROJECT_DIR / "artifcats"
DATASET_DIR = Path.cwd() / "dataset"

# Ventanas HOG
WIN_SIZE_LATERAL = (128, 64)   # Aspect ratio 2.0 (camión largo/tráiler)
WIN_SIZE_FRONTAL = (96, 96)    # Aspect ratio 1.0 (frontal cuadrado)

PAD_RATIO = 0.12
MIN_BOX_SIDE = 70              # Solo camiones grandes y nítidos
ASPECT_RANGE_LATERAL = (1.6, 3.2)
ASPECT_RANGE_FRONTAL = (0.75, 1.25)
BLUR_MIN_VAR = 80.0

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Configurar directorios de salida
POS_RAW_LAT_DIR = DATASET_DIR / "positives_raw_lateral"
POS_RAW_FRONT_DIR = DATASET_DIR / "positives_raw_frontal"
NEG_RAW_LAT_DIR = DATASET_DIR / "negatives_raw_lateral"
NEG_RAW_FRONT_DIR = DATASET_DIR / "negatives_raw_frontal"

for d in (POS_RAW_LAT_DIR, POS_RAW_FRONT_DIR, NEG_RAW_LAT_DIR, NEG_RAW_FRONT_DIR):
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)

# 1. Extracción de datasets
def extract_if_needed(archive_path: Path, extract_to: Path, marker_dir: Path):
    if not archive_path.exists():
        print(f"{archive_path.name}: no encontrado en artifcats/, se omite")
        return
    if marker_dir.exists() and any(marker_dir.iterdir()):
        print(f"{marker_dir.name}: ya extraído, se omite")
        return
    print(f"extrayendo {archive_path.name}...")
    extract_to.mkdir(parents=True, exist_ok=True)
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(extract_to)
    elif archive_path.suffix == ".tar":
        with tarfile.open(archive_path) as tf:
            tf.extractall(extract_to)
    print(f"{archive_path.name} -> {extract_to}")

extract_if_needed(ARTIFACTS_DIR / "Vehicles-coco.coco.zip", ARTIFACTS_DIR / "Vehicles-coco.coco", ARTIFACTS_DIR / "Vehicles-coco.coco")
extract_if_needed(ARTIFACTS_DIR / "CCU-Truck.coco.zip", ARTIFACTS_DIR / "CCU-Truck.coco", ARTIFACTS_DIR / "CCU-Truck.coco")
extract_if_needed(ARTIFACTS_DIR / "truck detection.coco.zip", ARTIFACTS_DIR / "Truck-detection.coco", ARTIFACTS_DIR / "Truck-detection.coco")
extract_if_needed(ARTIFACTS_DIR / "Truck.coco.zip", ARTIFACTS_DIR / "Truck.coco", ARTIFACTS_DIR / "Truck.coco")
extract_if_needed(ARTIFACTS_DIR / "val2017.zip", ARTIFACTS_DIR, ARTIFACTS_DIR / "val2017")
extract_if_needed(ARTIFACTS_DIR / "indoorCVPR_09.tar", ARTIFACTS_DIR / "MIT67_Indoor", ARTIFACTS_DIR / "MIT67_Indoor" / "Images")
extract_if_needed(ARTIFACTS_DIR / "places365_val_256.tar", ARTIFACTS_DIR / "Places365", ARTIFACTS_DIR / "Places365")
extract_if_needed(ARTIFACTS_DIR / "Billboard Detector.v2i.coco.zip", ARTIFACTS_DIR / "Billboard Detector.v2i.coco", ARTIFACTS_DIR / "Billboard Detector.v2i.coco")
extract_if_needed(ARTIFACTS_DIR / "Self Driving Car.v2-fixed-large.coco.zip", ARTIFACTS_DIR / "Self Driving Car.v2-fixed-large.coco", ARTIFACTS_DIR / "Self Driving Car.v2-fixed-large.coco")
extract_if_needed(ARTIFACTS_DIR / "Citypersons.coco.zip", ARTIFACTS_DIR / "Citypersons.coco", ARTIFACTS_DIR / "Citypersons.coco")

# 2. Utilidades de recorte y hash
def crop_with_padding(img, x, y, w, h, pad_ratio):
    ih, iw = img.shape[:2]
    pad_w, pad_h = w * pad_ratio, h * pad_ratio
    x0 = max(0, int(x - pad_w))
    y0 = max(0, int(y - pad_h))
    x1 = min(iw, int(x + w + pad_w))
    y1 = min(ih, int(y + h + pad_h))
    if x1 <= x0 or y1 <= y0:
        return None
    return img[y0:y1, x0:x1]

def perceptual_hash(gray_crop):
    small = cv2.resize(gray_crop, (8, 8))
    return (small > small.mean()).tobytes()

# 3. Extracción de Positivos de Camión Grande
CCU_LARGE_CLASSES = ["tractor_unit", "dump_truck", "concrete_mixer_truck", "tanker_truck", "tow_truck"]

def extract_positives_from_coco(dataset_dir: Path, splits, categories, out_dir: Path, tag: str,
                                 win_size: tuple, aspect_range: tuple, seen_hashes: set):
    n_saved = n_dup = n_blur = 0
    for split in splits:
        split_dir = dataset_dir / split
        ann_path = split_dir / "_annotations.coco.json"
        if not ann_path.exists():
            continue
        coco = json.loads(ann_path.read_text())
        images_by_id = {im["id"]: im for im in coco["images"]}
        
        # Mapear nombres a IDs
        cat_ids = {c["id"] for c in coco["categories"] if c["name"] in categories}
        
        # Para CCU-Truck, también podemos aceptar box_truck y flatbed_truck si miden de ancho >= 85px
        is_ccu = (tag == "ccu")
        ccu_medium_ids = set()
        if is_ccu:
            ccu_medium_ids = {c["id"] for c in coco["categories"] if c["name"] in ["box_truck", "flatbed_truck"]}

        for ann in coco["annotations"]:
            is_medium_ccu_large = (ann["category_id"] in ccu_medium_ids and max(ann["bbox"][2], ann["bbox"][3]) >= 85)
            if ann["category_id"] not in cat_ids and not is_medium_ccu_large:
                continue
                
            im_info = images_by_id.get(ann["image_id"])
            if im_info is None:
                continue
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            if min(w, h) < MIN_BOX_SIDE:
                continue
            if aspect_range is not None and not (aspect_range[0] <= w / h <= aspect_range[1]):
                continue
                
            img_path = split_dir / im_info["file_name"]
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            crop = crop_with_padding(img, x, y, w, h, PAD_RATIO)
            if crop is None or crop.size == 0:
                continue
                
            crop_resized = cv2.resize(crop, win_size)
            gray = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY)
            if BLUR_MIN_VAR > 0 and cv2.Laplacian(gray, cv2.CV_64F).var() < BLUR_MIN_VAR:
                n_blur += 1
                continue
            if seen_hashes is not None:
                hsh = perceptual_hash(gray)
                if hsh in seen_hashes:
                    n_dup += 1
                    continue
                seen_hashes.add(hsh)
                
            out_path = out_dir / f"{tag}_{split}_{ann['id']}.jpg"
            cv2.imwrite(str(out_path), crop_resized)
            n_saved += 1
            
    print(f"  [{tag}] positivos guardados: {n_saved} (descartados: {n_dup} dups, {n_blur} borrosos)")
    return n_saved

# Fuentes de positivos
STANDARD_SPLITS = ["train", "valid", "test"]
pos_sources = [
    (ARTIFACTS_DIR / "V.coco", STANDARD_SPLITS, {"truck"}, "vcoco"),
    (ARTIFACTS_DIR / "CCU-Truck.coco", STANDARD_SPLITS, set(CCU_LARGE_CLASSES), "ccu"),
    (ARTIFACTS_DIR / "Truck.coco", STANDARD_SPLITS, {"truck"}, "truckcoco"),
    (ARTIFACTS_DIR / "Truck-detection.coco", STANDARD_SPLITS, {"truck"}, "truckdet"),
    (ARTIFACTS_DIR / "Vehicles-coco.coco", STANDARD_SPLITS, {"truck"}, "vehcoco"),
]

print("--- Extrayendo positivos LATERAL (Aspect Ratio 2.0) ---")
seen_hashes_lat = set()
total_pos_lat = 0
for src_dir, splits, cats, tag in pos_sources:
    total_pos_lat += extract_positives_from_coco(
        src_dir, splits, cats, POS_RAW_LAT_DIR, tag,
        WIN_SIZE_LATERAL, ASPECT_RANGE_LATERAL, seen_hashes_lat
    )
print(f"TOTAL positivos lateral: {total_pos_lat}")

print("\n--- Extrayendo positivos FRONTAL (Aspect Ratio 1.0) ---")
seen_hashes_front = set()
total_pos_front = 0
for src_dir, splits, cats, tag in pos_sources:
    total_pos_front += extract_positives_from_coco(
        src_dir, splits, cats, POS_RAW_FRONT_DIR, tag,
        WIN_SIZE_FRONTAL, ASPECT_RANGE_FRONTAL, seen_hashes_front
    )
print(f"TOTAL positivos frontal: {total_pos_front}")


# 4. Extracción de Negativos Estructurados
def extract_negatives_from_coco(dataset_dir: Path, splits, categories, out_dir: Path, tag: str,
                                 win_size: tuple, max_crops: int, extract_wheels=False):
    n_saved = 0
    # Guardar crops de llantas para evitar FP en llantas
    n_wheels_saved = 0
    
    for split in splits:
        if n_saved >= max_crops:
            break
        split_dir = dataset_dir / split
        ann_path = split_dir / "_annotations.coco.json"
        if not ann_path.exists():
            continue
        coco = json.loads(ann_path.read_text())
        images_by_id = {im["id"]: im for im in coco["images"]}
        cat_ids = {c["id"] for c in coco["categories"] if c["name"] in categories}
        
        # Mezclar anotaciones para variedad
        annotations = coco["annotations"].copy()
        random.shuffle(annotations)
        
        for ann in annotations:
            if n_saved >= max_crops:
                break
            if ann["category_id"] not in cat_ids:
                continue
            im_info = images_by_id.get(ann["image_id"])
            if im_info is None:
                continue
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
                
            img_path = split_dir / im_info["file_name"]
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            # Crop del vehículo completo
            crop = crop_with_padding(img, x, y, w, h, 0.05)
            if crop is not None and crop.size > 0:
                crop_resized = cv2.resize(crop, win_size)
                out_path = out_dir / f"{tag}_neg_{split}_{ann['id']}.jpg"
                cv2.imwrite(str(out_path), crop_resized)
                n_saved += 1
                
            # Extracción explícita de la llanta (bottom 30%)
            if extract_wheels and n_wheels_saved < (max_crops // 2):
                wy = y + 0.7 * h
                wh = 0.3 * h
                wheel_crop = crop_with_padding(img, x, wy, w, wh, 0.05)
                if wheel_crop is not None and wheel_crop.size > 0:
                    wheel_resized = cv2.resize(wheel_crop, win_size)
                    out_path = out_dir / f"{tag}_wheel_neg_{split}_{ann['id']}.jpg"
                    cv2.imwrite(str(out_path), wheel_resized)
                    n_wheels_saved += 1
                    
    print(f"  [{tag}] negativos: {n_saved}, llantas: {n_wheels_saved}")
    return n_saved + n_wheels_saved

# Haremos negativos para ambas vistas por separado
for name, win_sz, neg_dir in [("lateral", WIN_SIZE_LATERAL, NEG_RAW_LAT_DIR), 
                               ("frontal", WIN_SIZE_FRONTAL, NEG_RAW_FRONT_DIR)]:
    print(f"\n--- Extrayendo negativos para vista {name.upper()} ---")
    
    # 1. Buses (3,000 crops)
    n_buses = extract_negatives_from_coco(
        ARTIFACTS_DIR / "Vehicles-coco.coco", STANDARD_SPLITS, {"bus"}, neg_dir, "bus_vehcoco",
        win_sz, max_crops=2000, extract_wheels=True
    )
    n_buses += extract_negatives_from_coco(
        ARTIFACTS_DIR / "CCU-Truck.coco", STANDARD_SPLITS, {"bus"}, neg_dir, "bus_ccu",
        win_sz, max_crops=1000, extract_wheels=True
    )
    
    # 2. Autos (3,000 crops)
    n_cars = extract_negatives_from_coco(
        ARTIFACTS_DIR / "Vehicles-coco.coco", STANDARD_SPLITS, {"car", "motorcycle"}, neg_dir, "car_vehcoco",
        win_sz, max_crops=1500, extract_wheels=True
    )
    n_cars += extract_negatives_from_coco(
        ARTIFACTS_DIR / "CCU-Truck.coco", STANDARD_SPLITS, {"car"}, neg_dir, "car_ccu",
        win_sz, max_crops=1000, extract_wheels=True
    )
    n_cars += extract_negatives_from_coco(
        ARTIFACTS_DIR / "Self Driving Car.v2-fixed-large.coco", ["export"], {"car"}, neg_dir, "car_udacity",
        win_sz, max_crops=1000, extract_wheels=True
    )
    
    # 3. Carteles (Billboard)
    n_bill = extract_negatives_from_coco(
        ARTIFACTS_DIR / "Billboard Detector.v2i.coco", STANDARD_SPLITS, {"Billboard"}, neg_dir, "billboard",
        win_sz, max_crops=100
    )
    
    # 4. Señales y semáforos de Udacity
    n_signals = extract_negatives_from_coco(
        ARTIFACTS_DIR / "Self Driving Car.v2-fixed-large.coco", ["export"], 
        {"trafficLight", "biker", "pedestrian"}, neg_dir, "signal_udacity",
        win_sz, max_crops=1000
    )

    # 5. Escenas y parches aleatorios (TipoCasa, Citypersons, MIT67, Places365, COCO val2017, Webcam)
    def random_patch_aspect(img, target_w, target_h):
        ih, iw = img.shape[:2]
        aspect = target_w / target_h
        max_h = min(ih, int(iw / aspect))
        if max_h < 24:
            return None
        min_h = max(20, int(max_h * 0.35))
        h = random.randint(min_h, max_h)
        w = int(h * aspect)
        if w > iw or h > ih:
            return None
        x0 = random.randint(0, iw - w)
        y0 = random.randint(0, ih - h)
        return img[y0:y0 + h, x0:x0 + w]

    def crop_random_patches(image_paths, out_dir: Path, tag: str, patches_per_image=1, max_patches=None):
        n_saved = 0
        random.shuffle(image_paths)
        for img_path in image_paths:
            if max_patches is not None and n_saved >= max_patches:
                break
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            for _ in range(patches_per_image):
                patch = random_patch_aspect(img, win_sz[0], win_sz[1])
                if patch is None:
                    continue
                patch = cv2.resize(patch, win_sz)
                out_path = out_dir / f"{tag}_rand_{img_path.stem}_{n_saved}.jpg"
                cv2.imwrite(str(out_path), patch)
                n_saved += 1
        print(f"  [{tag}] parches aleatorios: {n_saved}")
        return n_saved

    n_cp = crop_random_patches(list((ARTIFACTS_DIR / "Citypersons.coco").rglob("*.png")), neg_dir, "citypersons", patches_per_image=2, max_patches=3000)
    n_coco = crop_random_patches(list((ARTIFACTS_DIR / "val2017").glob("*.jpg")), neg_dir, "coco_val", patches_per_image=1, max_patches=3000)
    n_mit = crop_random_patches(list((ARTIFACTS_DIR / "MIT67_Indoor" / "Images").rglob("*.jpg")), neg_dir, "mit67", patches_per_image=1, max_patches=3000)
    n_places = crop_random_patches(list((ARTIFACTS_DIR / "Places365").rglob("*.jpg")), neg_dir, "places365", patches_per_image=1, max_patches=3000)
    n_casa = crop_random_patches(list((ARTIFACTS_DIR / "TipoCasa_V2.v2i.folder").rglob("*.jpg")), neg_dir, "tipocasa", patches_per_image=2, max_patches=2400)
    n_web = crop_random_patches(list((ARTIFACTS_DIR / "webcam_negatives").glob("*.jpg")), neg_dir, "webcam", patches_per_image=2, max_patches=700)
    
    total_neg_view = len(list(neg_dir.glob("*.jpg")))
    print(f"TOTAL negativos para {name}: {total_neg_view}")

print("\n=== Dataset preparado con éxito ===")
