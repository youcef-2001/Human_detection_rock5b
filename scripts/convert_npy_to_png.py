import os
import numpy as np
import cv2



def convert_npy_to_png(npy_folder="dataset_npy/", images_folder = "dataset/images/train/" , labels_folder = "dataset/labels/train/"    ):
    # --- CONFIGURATION ---
    # npy_folder = "dataset_npy/"     
    # images_folder = "dataset/images/train/"     # Dossier de sortie pour les PNG (YOLO)
    # labels_folder = "dataset/labels/train/"     # Dossier de sortie pour les TXT (YOLO)

    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(labels_folder, exist_ok=True)

    # L'échelle de température globale (CRUCIAL pour YOLO)

    TEMP_MIN_GLOBALE = 5.0 
    TEMP_MAX_GLOBALE = 55.0

    TEMP_SEUIL = 31.0   # Frontière Humain / Objet chaud
    SCALE_FACTOR = 10   # Agrandissement 32x24 -> 320x240 pour YOLO

    # Compteurs
    stats = {"Humains": 0, "Objets Chauds": 0}

    for filename in os.listdir(npy_folder):
        if not filename.endswith(".npy"):
            continue

        filepath = os.path.join(npy_folder, filename)
        
        # 1. Charger les températures brutes et exactes
        img_np = np.load(filepath)
        
        # 2. Créer l'image pour YOLO (Normalisation fixe pour préserver les contrastes thermiques)
        # Les températures hors limites sont "coupées" (clip) pour rester entre 15 et 45
        img_clipped = np.clip(img_np, TEMP_MIN_GLOBALE, TEMP_MAX_GLOBALE)
        img_8u = ((img_clipped - TEMP_MIN_GLOBALE) / (TEMP_MAX_GLOBALE - TEMP_MIN_GLOBALE) * 255.0).astype(np.uint8)
        
        # 3. Trouver les objets (via la méthode d'Otsu sur l'image 8-bit)
        _, mask = cv2.threshold(img_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Préparer le fichier texte YOLO
        label_filename = filename.replace('.npy', '.txt')
        label_filepath = os.path.join(labels_folder, label_filename)
        
        with open(label_filepath, 'w') as f:
            for contour in contours:
                # Ignorer le bruit (moins de 2 pixels thermiques de surface)
                if cv2.contourArea(contour) < 1:
                    continue
                    
                x, y, w, h = cv2.boundingRect(contour)
                
                # --- LE SECRET EST ICI ---
                # On retourne lire la température MAX dans le fichier .npy brut !
                zone_brute = img_np[y:y+h, x:x+w]
                temp_max_reelle = np.max(zone_brute)
                
                if temp_max_reelle >= TEMP_SEUIL:
                    class_id = 1 # Objet chaud
                    stats["Objets Chauds"] += 1
                else:
                    class_id = 0 # Humain
                    stats["Humains"] += 1
                    
                # Coordonnées YOLO (normalisées sur la résolution d'origine 32x24)
                x_center = (x + w / 2.0) / 32.0
                y_center = (y + h / 2.0) / 24.0
                w_norm = w / 32.0
                h_norm = h / 24.0
                
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

        # 4. Sauvegarder l'image PNG pour YOLO
        # Agrandissement "Nearest" pour ne pas flouter les pixels thermiques
        large_img = cv2.resize(img_8u, (32 * SCALE_FACTOR, 24 * SCALE_FACTOR), interpolation=cv2.INTER_NEAREST)
        
        # YOLO préfère souvent des images sur 3 canaux (RGB), même si elles sont en niveaux de gris
        large_img_rgb = cv2.cvtColor(large_img, cv2.COLOR_GRAY2BGR)
        
        img_filename = filename.replace('.npy', '.png')
        cv2.imwrite(os.path.join(images_folder, img_filename), large_img_rgb)

    print("Traitement terminé ! Votre dataset YOLO est prêt.")
    print(f"Bilan des détections : {stats['Humains']} Humains, {stats['Objets Chauds']} Objets Chauds.")


if __name__ == "__main__":
    convert_npy_to_png()