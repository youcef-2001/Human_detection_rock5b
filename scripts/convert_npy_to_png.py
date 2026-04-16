import os
import re
import numpy as np
import cv2
from scipy.ndimage import label, generate_binary_structure, distance_transform_edt
from scipy import ndimage



def parse_filename_metadata(filename):
    """
    Extrait les métadonnées du nom de fichier.
    Format: frame_{temp_min:.1f}_{temp_max:.1f}_{nb_personnes}_{nb_points_chauds}_{frame_counter}.npy
    """
    pattern = r'frame_(\d+\.?\d*)_(\d+\.?\d*)_(\d+)_(\d+)_(\d+)\.npy'
    match = re.match(pattern, filename)
    
    if match:
        return {
            'temp_min': float(match.group(1)),
            'temp_max': float(match.group(2)),
            'nb_personnes': int(match.group(3)),
            'nb_points_chauds': int(match.group(4)),
            'frame_counter': int(match.group(5))
        }
    return None


def separate_overlapping_objects(mask, img_np, temp_seuil=35.0):
    """
    Sépare les objets qui se chevauchent en utilisant la transformée en distance et watershed.
    Retourne une liste de contours séparés avec leurs propriétés.
    """
    # Appliquer une érosion douce pour séparer les objets
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded = cv2.erode(mask, kernel, iterations=1)
    
    # Distance transform pour trouver les "centres" des objets
    dist = cv2.distanceTransform(eroded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    
    # Normaliser la distance
    dist_norm = np.zeros_like(dist, dtype=np.uint8)
    cv2.normalize(dist, dist_norm, 0, 255, cv2.NORM_MINMAX)
    
    # Trouver les marqueurs (pics de distance)
    _, markers = cv2.threshold(dist_norm, 0.7 * dist_norm.max(), 255, cv2.THRESH_BINARY)
    markers = cv2.morphologyEx(markers, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Labéliser les marqueurs
    num_labels, markers_labeled = cv2.connectedComponents(markers.astype(np.uint8))
    
    # Appliquer watershed pour une meilleure séparation
    markers_watershed = cv2.watershed(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), markers_labeled)
    
    # Extraire les contours séparés
    separated_regions = []
    for label_id in range(1, num_labels):
        region_mask = (markers_watershed == label_id).astype(np.uint8) * 255
        
        if cv2.countNonZero(region_mask) < 2:  # Ignorer les régions trop petites
            continue
        
        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = contours[0]
            separated_regions.append({
                'contour': contour,
                'mask': region_mask,
                'label_id': label_id
            })
    
    return separated_regions if separated_regions else []


def compute_confidence_score(zone_thermique, img_np_raw, temp_min_frame, temp_max_frame, temp_seuil=35.0):
    """
    Calcule un score de confiance basé sur les statistiques thermiques.
    """
    temp_mean = np.mean(zone_thermique)
    temp_std = np.std(zone_thermique)
    temp_max = np.max(zone_thermique)
    temp_range = temp_max_frame - temp_min_frame
    human_temp_min = 24.0
    
    if temp_range < 1:
        temp_range = 1
    
    # Normaliser les statistiques
    normalized_mean = (temp_mean - temp_min_frame) / temp_range
    contrast = temp_std / (temp_range if temp_range > 0 else 1)
    
    # Score basé sur la distinction thermique
    if temp_max >= temp_seuil:
        # Objet chaud: score élevé si bien au-dessus du seuil
        class_type = 1
        confidence = min(1.0, (temp_max - temp_seuil) / (temp_max_frame - temp_seuil + 1e-6))
    elif temp_max < human_temp_min:
        # Trop froid pour être un humain: on rejette la zone
        return None, 0.0
    else:
        # Humain: score élevé si proche du seuil par le bas
        class_type = 0
        confidence = min(1.0, (temp_max - human_temp_min) / (temp_seuil - human_temp_min + 1e-6))
    
    # Pénalité si faible contraste thermique
    if contrast < 0.05:
        confidence *= 0.7
    
    return class_type, confidence


def convert_npy_to_png(npy_folder="dataset_npy/", images_folder = "dataset/images/train/" , labels_folder = "dataset/labels/train/", confidence_threshold=0.3):
    # --- CONFIGURATION ---
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(labels_folder, exist_ok=True)

    # L'échelle de température globale (CRUCIAL pour YOLO)
    TEMP_MIN_GLOBALE = 5.0 
    TEMP_MAX_GLOBALE = 60.0
    TEMP_SEUIL = 35.0   # Frontière Humain / Objet chaud
    SCALE_FACTOR = 10   # Agrandissement 32x24 -> 320x240 pour YOLO

    # Compteurs
    stats = {
        "Humains_detectes": 0, 
        "Objets_Chauds_detectes": 0,
        "Humains_attendus": 0,
        "Objets_Chauds_attendus": 0,
        "fichiers_traites": 0
    }

    for filename in sorted(os.listdir(npy_folder)):
        if not filename.endswith(".npy"):
            continue

        filepath = os.path.join(npy_folder, filename)
        
        # Extraire les métadonnées du nom de fichier
        metadata = parse_filename_metadata(filename)
        
        if not metadata:
            print(f"⚠️  Format de nom invalide: {filename}")
            continue
        
        expected_humans = metadata['nb_personnes']
        expected_hot_objects = metadata['nb_points_chauds']
        temp_min_frame = metadata['temp_min']
        temp_max_frame = metadata['temp_max']
        
        stats["Humains_attendus"] += expected_humans
        stats["Objets_Chauds_attendus"] += expected_hot_objects
        stats["fichiers_traites"] += 1
        
        # 1. Charger les températures brutes
        img_np = np.load(filepath)
        
        # 2. Créer l'image normalisée (utiliser les limites du frame si disponibles)
        img_clipped = np.clip(img_np, TEMP_MIN_GLOBALE, TEMP_MAX_GLOBALE)
        img_8u = ((img_clipped - TEMP_MIN_GLOBALE) / (TEMP_MAX_GLOBALE - TEMP_MIN_GLOBALE) * 255.0).astype(np.uint8)
        
        # 3. Segmentation robuste avec Otsu
        _, mask = cv2.threshold(img_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 4. Séparation des objets qui se chevauchent
        separated_regions = separate_overlapping_objects(mask, img_np, TEMP_SEUIL)
        
        if not separated_regions:
            # Fallback: utiliser les contours standards si la séparation échoue
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            separated_regions = [{'contour': c, 'mask': None} for c in contours]
        
        # Préparer le fichier texte YOLO
        label_filename = filename.replace('.npy', '.txt')
        label_filepath = os.path.join(labels_folder, label_filename)
        
        detections = []
        
        for region in separated_regions:
            contour = region.get('contour')
            if contour is None:
                continue
            
            area = cv2.contourArea(contour)
            
            # Ignorer le bruit très petit
            if area < 2:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # Vérifier les limites
            if y < 0 or y + h > img_np.shape[0] or x < 0 or x + w > img_np.shape[1]:
                continue
            
            zone_brute = img_np[y:y+h, x:x+w]
            
            # Calculer le score de confiance et la classe
            class_id, confidence = compute_confidence_score(
                zone_brute, img_np, temp_min_frame, temp_max_frame, TEMP_SEUIL
            )

            if class_id is None:
                continue
            
            # Filtrer par confiance
            if confidence < confidence_threshold:
                continue
            
            # Coordonnées YOLO (normalisées sur la résolution d'origine 32x24)
            x_center = (x + w / 2.0) / 32.0
            y_center = (y + h / 2.0) / 24.0
            w_norm = w / 32.0
            h_norm = h / 24.0
            
            # Clamp les valeurs YOLO
            x_center = np.clip(x_center, 0.0, 1.0)
            y_center = np.clip(y_center, 0.0, 1.0)
            w_norm = np.clip(w_norm, 0.0, 1.0)
            h_norm = np.clip(h_norm, 0.0, 1.0)
            
            detections.append({
                'class_id': class_id,
                'confidence': confidence,
                'x_center': x_center,
                'y_center': y_center,
                'w_norm': w_norm,
                'h_norm': h_norm
            })
            
            if class_id == 0:
                stats["Humains_detectes"] += 1
            else:
                stats["Objets_Chauds_detectes"] += 1
        
        # Écrire les détections dans le fichier label
        with open(label_filepath, 'w') as f:
            for det in detections:
                f.write(f"{det['class_id']} {det['x_center']:.6f} {det['y_center']:.6f} {det['w_norm']:.6f} {det['h_norm']:.6f}\n")

        # 5. Sauvegarder l'image PNG
        large_img = cv2.resize(img_8u, (32 * SCALE_FACTOR, 24 * SCALE_FACTOR), interpolation=cv2.INTER_NEAREST)
        large_img_rgb = cv2.cvtColor(large_img, cv2.COLOR_GRAY2BGR)
        
        img_filename = filename.replace('.npy', '.png')
        cv2.imwrite(os.path.join(images_folder, img_filename), large_img_rgb)

    # Afficher les statistiques complètes
    print("\n" + "="*60)
    print("✅ Traitement terminé ! Dataset YOLO prêt.")
    print("="*60)
    print(f"📊 Fichiers traités: {stats['fichiers_traites']}")
    print(f"\n📈 Humains:")
    print(f"   Détectés: {stats['Humains_detectes']}")
    print(f"   Attendus: {stats['Humains_attendus']}")
    print(f"   Taux: {stats['Humains_detectes']/max(1, stats['Humains_attendus'])*100:.1f}%")
    print(f"\n🔥 Objets chauds:")
    print(f"   Détectés: {stats['Objets_Chauds_detectes']}")
    print(f"   Attendus: {stats['Objets_Chauds_attendus']}")
    print(f"   Taux: {stats['Objets_Chauds_detectes']/max(1, stats['Objets_Chauds_attendus'])*100:.1f}%")
    print("="*60)


if __name__ == "__main__":
    # Paramètres ajustables:
    # - npy_folder: chemin vers les fichiers .npy
    # - images_folder: où sauvegarder les PNG
    # - labels_folder: où sauvegarder les annotations YOLO
    # - confidence_threshold: score minimum pour inclure une détection (0.0-1.0)
    convert_npy_to_png(
        npy_folder="dataset_npy/",
        images_folder="dataset/images/train/",
        labels_folder="dataset/labels/train/",
        confidence_threshold=0.001  # Ajustez pour plus/moins de détections
    )