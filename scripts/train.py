from ultralytics import YOLO
import os
import locale

def _setup_utf8():
    # Variables locales UTF-8
    os.environ.setdefault("LANG", "C.UTF-8")
    os.environ.setdefault("LC_ALL", "C.UTF-8")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    # Recharge la locale depuis les variables d'environnement
    try:
        locale.setlocale(locale.LC_ALL, "")
    except locale.Error:
        pass

def main():
    _setup_utf8()

    print("=== Démarrage de l'entraînement YOLOv8 Thermique ===")
    
    # 1. Charger le modèle YOLOv8 nano (le plus léger et rapide, parfait pour le NPU du RK3588)
    model = YOLO('yolov8n.pt')

    # 2. Lancer l'entraînement
    results = model.train(
        data='./dataset/data.yaml', # <-- Vérifiez que ce chemin correspond à votre fichier yaml
        epochs=300,                 # 100 passages sur vos 2000 images est un excellent point de départ
        imgsz=320,                  # La résolution à laquelle nous avons agrandi les images
        batch=16,                   # Mettez 32 si votre PC a beaucoup de RAM, sinon gardez 16
        device='0',               # Mettez '0' si vous avez une carte graphique Nvidia sur votre PC
        project='models',
        name='mon_YOLOv8',
        
        # --- OPTIMISATIONS THERMIQUES STRICTES (Ne pas modifier) ---
        # On interdit à YOLO de modifier les couleurs et la luminosité 
        # pour que les températures (niveaux de gris) ne soient pas faussées.
        # hsv_h=0.0,             
        # hsv_s=0.0,             
        # hsv_v=0.0,             
        
        # --- AUGMENTATION GÉOMÉTRIQUE MODÉRÉE ---
        # On aide le modèle à être plus robuste en créant de fausses situations
        degrees=10.0,          # Rotation légère des images (+/- 10 degrés)
        translate=0.1,         # Décalage de l'image de 10%
        scale=0.1,             # Zoom léger (10%)
        fliplr=0.5,            # Effet miroir gauche/droite à 50% de chance
        flipud=0.0,            # Pas de miroir vertical (les humains ne sont pas à l'envers)
        mosaic=1.0             # Mélange 4 images en 1 (excellent pour les petits objets)
    )

    print("\n=== Entraînement terminé avec succès ! ===")
    print("Votre modèle final est sauvegardé ici :")
    print("modeles_thermiques/v_finale_2classes/weights/best.pt")

if __name__ == '__main__':
    # Ce bloc if __name__ == '__main__' est parfois requis par Windows/PyTorch 
    # pour bien gérer le multi-threading pendant l'entraînement.
    main()