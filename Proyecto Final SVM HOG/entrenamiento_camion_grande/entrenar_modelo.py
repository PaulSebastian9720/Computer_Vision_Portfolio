import gc
import random
from pathlib import Path
import cv2
import numpy as np
import albumentations as A
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

# Directorios
DATASET_DIR = Path.cwd() / "dataset"
ARTIFACTS_DIR = Path.cwd().parent / "artifcats"

# Configuración
VIEWS = {
    "frontal": {
        "win_size": (96, 96),
        "pos_raw_dir": DATASET_DIR / "positives_raw_frontal",
        "neg_raw_dir": DATASET_DIR / "negatives_raw_frontal",
        "pos_aug_dir": DATASET_DIR / "positives_aug_frontal",
        "neg_aug_dir": DATASET_DIR / "negatives_aug_frontal",
        "out_yml": DATASET_DIR / "detector_frontal.yml",
        "mining_win_stride": (16, 16),
        "target_per_class": 22000,
    },
    "lateral": {
        "win_size": (128, 64),
        "pos_raw_dir": DATASET_DIR / "positives_raw_lateral",
        "neg_raw_dir": DATASET_DIR / "negatives_raw_lateral",
        "pos_aug_dir": DATASET_DIR / "positives_aug_lateral",
        "neg_aug_dir": DATASET_DIR / "negatives_aug_lateral",
        "out_yml": DATASET_DIR / "detector_lateral.yml",
        "mining_win_stride": (24, 16),
        "target_per_class": 22000,
    }
}

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

C_VALUES = [0.001, 0.01, 0.1, 1.0]
VAL_FRACTION = 0.15
MAX_ITER = 25000

MINING_ROUNDS = 4
MAX_HARD_NEG_PER_ROUND = 4000
MIN_HARD_TO_CONTINUE = 400
POOL_SAMPLE_PER_ROUND = 8000
MINING_THRESHOLD = -0.15

# Augmentations
augment_pos = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
    A.Affine(scale=(0.95, 1.05), p=0.3),
    A.MotionBlur(blur_limit=3, p=0.15),
])

augment_neg = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.7, border_mode=cv2.BORDER_REPLICATE),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
    A.MotionBlur(blur_limit=5, p=0.3),
    A.GaussNoise(p=0.3),
    A.Affine(scale=(0.85, 1.15), p=0.5),
])

def expand_to_target(raw_dir: Path, out_dir: Path, target: int, transform):
    raw_paths = sorted(raw_dir.glob("*.jpg"))
    if not raw_paths:
        print(f"  [Error] sin imágenes base en {raw_dir}")
        return 0
    
    # Limpiar y crear
    if out_dir.exists():
        shutil = __import__('shutil')
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    n_saved = 0
    # Guardar originales
    for p in raw_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        cv2.imwrite(str(out_dir / p.name), img)
        n_saved += 1
        
    # Aumentar
    idx = 0
    while n_saved < target:
        src = raw_paths[idx % len(raw_paths)]
        img = cv2.imread(str(src))
        if img is None:
            idx += 1
            continue
        aug_img = transform(image=img)["image"]
        out_path = out_dir / f"{src.stem}_aug{idx}.jpg"
        cv2.imwrite(str(out_path), aug_img)
        n_saved += 1
        idx += 1
    return n_saved

def balance_dataset(X_feat, y_labels):
    pos_idx = np.where(y_labels == 1)[0]
    neg_idx = np.where(y_labels == -1)[0]
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)
    if abs(n_pos - n_neg) <= 5:
        return X_feat, y_labels
    if n_pos < n_neg:
        extra_idx = np.random.choice(pos_idx, n_neg - n_pos, replace=True)
        new_pos_idx = np.concatenate([pos_idx, extra_idx])
        new_idx = np.concatenate([new_pos_idx, neg_idx])
    else:
        extra_idx = np.random.choice(neg_idx, n_pos - n_neg, replace=True)
        new_neg_idx = np.concatenate([neg_idx, extra_idx])
        new_idx = np.concatenate([pos_idx, new_neg_idx])
    np.random.shuffle(new_idx)
    return X_feat[new_idx], y_labels[new_idx]

# Pool de imágenes para mining
mit67_paths = sorted((ARTIFACTS_DIR / "MIT67_Indoor" / "Images").rglob("*.jpg"))
places_paths = sorted((ARTIFACTS_DIR / "Places365").rglob("*.jpg")) if (ARTIFACTS_DIR / "Places365").exists() else []
billboard_paths = sorted((ARTIFACTS_DIR / "Billboard Detector.v2i.coco").rglob("*.jpg")) if (ARTIFACTS_DIR / "Billboard Detector.v2i.coco").exists() else []
webcam_paths = sorted((ARTIFACTS_DIR / "webcam_negatives").glob("*.jpg")) if (ARTIFACTS_DIR / "webcam_negatives").exists() else []

# COCO val2017 sin vehículos
coco_clean_paths = []
val_dir = ARTIFACTS_DIR / "val2017"
ann_zip = ARTIFACTS_DIR / "annotations_trainval2017.zip"
if ann_zip.exists() and val_dir.exists():
    import zipfile
    import json
    with zipfile.ZipFile(ann_zip) as zf:
        with zf.open("annotations/instances_val2017.json") as f:
            cocoval = json.load(f)
    vehicle_cats = {c["id"] for c in cocoval["categories"] if c["name"] in ["truck", "car", "bus", "motorcycle"]}
    img_ids_with_vehicles = {a["image_id"] for a in cocoval["annotations"] if a["category_id"] in vehicle_cats}
    coco_clean_paths = [val_dir / im["file_name"] for im in cocoval["images"]
                        if im["id"] not in img_ids_with_vehicles and (val_dir / im["file_name"]).exists()]

mining_pool = mit67_paths + places_paths + coco_clean_paths
mining_always = webcam_paths + billboard_paths

print(f"Fuentes para Mining: Pool={len(mining_pool)}, Obligatorias={len(mining_always)}")

def mine_hard_negatives(clf, hog, win_size, image_paths, win_stride, cap_total):
    mhog = cv2.HOGDescriptor(win_size, (16, 16), (8, 8), (8, 8), 9)
    mhog.setSVMDetector(np.append(clf.coef_[0].astype(np.float32), np.float32(clf.intercept_[0])))
    feats = []
    
    for img_path in image_paths:
        if len(feats) >= cap_total:
            break
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None or min(img.shape[:2]) < min(win_size):
            continue
        if max(img.shape[:2]) > 640:
            s = 640 / max(img.shape[:2])
            img = cv2.resize(img, None, fx=s, fy=s)
            
        boxes, _ = mhog.detectMultiScale(img, hitThreshold=MINING_THRESHOLD,
                                         winStride=win_stride, scale=1.2)
        for (bx, by, bw, bh) in boxes:
            patch = img[max(0, by):by + bh, max(0, bx):bx + bw]
            if patch.size == 0:
                continue
            feats.append(hog.compute(cv2.resize(patch, win_size)).flatten())
            
    return np.array(feats[:cap_total], dtype=np.float32)

def train_view(name, cfg):
    win_size = cfg["win_size"]
    print(f"\n==========================================")
    print(f"ENTRENANDO VISTA: {name.upper()} ({win_size[0]}x{win_size[1]})")
    print(f"==========================================")
    
    # 1. Augmentation
    print("Expandiendo y aumentando dataset...")
    n_raw_negs = len(list(cfg["neg_raw_dir"].glob("*.jpg")))
    target_sz = max(cfg["target_per_class"], n_raw_negs)
    n_pos = expand_to_target(cfg["pos_raw_dir"], cfg["pos_aug_dir"], target_sz, augment_pos)
    n_neg = expand_to_target(cfg["neg_raw_dir"], cfg["neg_aug_dir"], target_sz, augment_neg)
    print(f"  Positivos aumentados: {n_pos} | Negativos aumentados: {n_neg}")
    
    # 2. Descriptores HOG
    hog = cv2.HOGDescriptor(win_size, (16, 16), (8, 8), (8, 8), 9)
    
    def compute_features(img_dir):
        feats = []
        paths = sorted(img_dir.glob("*.jpg"))
        for p in paths:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if img.shape[:2] != (win_size[1], win_size[0]):
                img = cv2.resize(img, win_size)
            feats.append(hog.compute(img).flatten())
        return np.array(feats, dtype=np.float32)
        
    print("Extrayendo descriptores HOG de positivos...")
    X_pos = compute_features(cfg["pos_aug_dir"])
    print("Extrayendo descriptores HOG de negativos...")
    X_neg = compute_features(cfg["neg_aug_dir"])
    
    X = np.vstack([X_pos, X_neg])
    y = np.hstack([np.ones(len(X_pos), dtype=np.int32), -np.ones(len(X_neg), dtype=np.int32)])
    print(f"Dataset X original: {X.shape}")
    
    # 3. Barrido de C
    print("Barrido del hiperparámetro C...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=VAL_FRACTION, random_state=SEED, stratify=y)
    X_train, y_train = balance_dataset(X_train, y_train)
    X_val, y_val = balance_dataset(X_val, y_val)
    
    results = []
    for c in C_VALUES:
        clf_c = LinearSVC(C=c, max_iter=MAX_ITER, dual="auto", class_weight="balanced")
        clf_c.fit(X_train, y_train)
        val_acc = clf_c.score(X_val, y_val)
        results.append((c, val_acc))
        print(f"  C={c:<8} val_acc={val_acc:.4f}")
        
    best_c = max(results, key=lambda r: r[1])[0]
    print(f"Mejor C seleccionado: {best_c}")
    
    del X_train, X_val, y_train, y_val
    gc.collect()
    
    # Entrenar modelo inicial con C óptimo
    X_bal, y_bal = balance_dataset(X, y)
    clf = LinearSVC(C=best_c, max_iter=MAX_ITER, dual="auto", class_weight="balanced")
    clf.fit(X_bal, y_bal)
    print(f"Accuracy inicial pre-mining: {clf.score(X, y):.4f}")
    
    # 4. Hard-Negative Mining Loop
    X_acc, y_acc = X, y
    for rnd in range(1, MINING_ROUNDS + 1):
        print(f"Ronda de Mining {rnd}/{MINING_ROUNDS}...")
        # Selección aleatoria de fuentes de mining
        sources = mining_always + random.sample(mining_pool, min(POOL_SAMPLE_PER_ROUND, len(mining_pool)))
        
        # Minar
        hard = mine_hard_negatives(clf, hog, win_size, sources, cfg["mining_win_stride"], MAX_HARD_NEG_PER_ROUND)
        n_hard = len(hard)
        print(f"  Ronda {rnd}: minados {n_hard} falsos positivos difíciles")
        
        if n_hard == 0:
            print("  No hay más negativos difíciles. Convergencia alcanzada.")
            break
            
        # Acumular
        X_acc = np.vstack([X_acc, hard])
        y_acc = np.hstack([y_acc, -np.ones(n_hard, dtype=np.int32)])
        
        # Balancear 1:1 estrictamente
        X_acc_bal, y_acc_bal = balance_dataset(X_acc, y_acc)
        
        # Reentrenar
        clf = LinearSVC(C=best_c, max_iter=MAX_ITER, dual="auto", class_weight="balanced")
        clf.fit(X_acc_bal, y_acc_bal)
        
        print(f"  Reentrenado con {X_acc_bal.shape[0]} muestras. Accuracy en acumulado: {clf.score(X_acc, y_acc):.4f}")
        
        del hard, X_acc_bal, y_acc_bal
        gc.collect()
        
        if n_hard < MIN_HARD_TO_CONTINUE:
            print("  Mining converge (pocos negativos nuevos). Deteniendo.")
            break
            
    # 5. Verificación de signo e intercept
    w = clf.coef_[0].astype(np.float32)
    b = np.float32(clf.intercept_[0])
    
    # Muestras aleatorias para calibrar umbral
    sp = X_pos[np.random.choice(len(X_pos), min(200, len(X_pos)), replace=False)]
    sn = X_neg[np.random.choice(len(X_neg), min(200, len(X_neg)), replace=False)]
    sp_s = sp @ w + b
    sn_s = sn @ w + b
    
    print(f"Score Positivos (media): {sp_s.mean():.3f} | Score Negativos (media): {sn_s.mean():.3f}")
    assert sp_s.mean() > sn_s.mean(), "Signo del SVM invertido!"
    
    threshold = (sp_s.mean() + sn_s.mean()) / 2
    print(f"Umbral sugerido (hitThreshold): {threshold:.3f}")
    
    # 6. Guardar en formato YML de OpenCV
    detector_vector = np.append(w, b)
    fs = cv2.FileStorage(str(cfg["out_yml"]), cv2.FileStorage_WRITE)
    fs.write("win_width", win_size[0])
    fs.write("win_height", win_size[1])
    fs.write("svm_detector", detector_vector.reshape(1, -1))
    fs.write("threshold", float(threshold))
    fs.release()
    print(f"Modelo guardado en: {cfg['out_yml']}")
    
    # Limpieza final
    del X, y, X_pos, X_neg, X_acc, y_acc, clf, hog
    gc.collect()

# Entrenar ambas vistas
for view_name, view_cfg in VIEWS.items():
    train_view(view_name, view_cfg)

print("\n=== Todos los modelos de camión grande entrenados con éxito ===")
