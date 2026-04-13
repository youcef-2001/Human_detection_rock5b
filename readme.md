# Human_detection_rock5b

Détection de personnes à partir d’un capteur thermique basé sur MLX90640, avec entraînement YOLOv8 sur PC et déploiement optimisé sur ROCK 5B (modèles ONNX et RKNN).

## Objectifs

- Acquérir des frames thermiques depuis un capteur MLX90640 (caméra IR 32×24).  
- Préparer et annoter ces données pour entraîner un modèle YOLOv8 de détection de personnes.  
- Exporter le modèle entraîné vers ONNX puis RKNN pour l’exécuter sur la ROCK 5B.  
- Fournir des scripts pour l’inférence temps réel :
  - sur PC (CPU/GPU) ;
  - sur ROCK 5B (NPU + RKNN) ;
  - via un backend serveur (WebSocket / HTTP) autour du modèle RKNN.

## Prérequis

- Une carte Radxa ROCK 5B (ou équivalent) pour l’inférence embarquée.
- Un PC pour l’entraînement YOLOv8.
- Un capteur/caméra thermique MLX90640BAA ou compatible (32×24 pixels IR).  
- Python 3.9+ avec les bibliothèques listées dans `requirements.txt`.
- SDK RKNN/RKNPU correctement installé sur la ROCK 5B (pour l’inférence NPU).

## Installation

Cloner le dépôt :

```bash
git clone https://github.com/youcef-2001/Human_detection_rock5b.git
cd Human_detection_rock5b
```

(Optionnel) Créer et activer un environnement virtuel :

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows
# .venv\Scripts\activate
```

Installer les dépendances :

```bash
pip install -r requirements.txt
```

Installer et configurer le SDK RKNN / RKNPU sur la ROCK 5B selon la documentation du constructeur.

## Utilisation rapide

### Inférence sur PC

Exemple typique (à adapter à tes options réelles) :

```bash
python mainPC.py --weights runs/detect/models/mon_YOLOv8/weights/best.pt --source 0 --conf 0.5
```

- `--weights` : chemin vers un modèle YOLOv8 (`.pt` ou `.onnx`).
- `--source` : index de la caméra ou chemin vers une vidéo/fichier image.
- `--conf` : seuil de confiance minimal.

### Inférence sur ROCK 5B (local)

```bash
python main.py --model rknn/mon_YOLOv84.rknn --source 0 --conf 0.5
```

- `--model` : fichier RKNN compilé.
- `--source` : flux vidéo (caméra connectée à la ROCK 5B).
- `--conf` : seuil de confiance.

### Backend serveur RKNN

```bash
python server/backend_ws_rknn.py --model rknn/mon_YOLOv84.rknn --host 0.0.0.0 --port 8000
```

Un client externe (PC, navigateur, autre microcontrôleur) peut alors envoyer des images/frames au serveur pour obtenir les détections en réponse.

---

## Structure du projet

```text
Human_detection_rock5b/
├── .gitignore
├── main.py
├── mainPC.py
├── requirements.txt
├── Screenshot_*.png
├── screen_entrainement.png
│
├── dataset/
│   └── data.yaml
│
├── dataset_npy/
│   ├── frame_16.7_25.8_3_0_28.npy
│   ├── frame_16.7_25.8_3_0_42.npy
│   ├── frame_16.7_26.7_6_0_667.npy
│   ├── frame_16.8_25.3_3_0_24.npy
│   ├── frame_19.7_26.9_2_1151.npy
│   ├── frame_19.7_27.0_1_1155.npy
│   └── frame_19.7_27.8_2_1144.npy
│
├── onnx/
│   ├── best.onnx
│   ├── mon_YOLOv84.onnx
│   └── mon_YOLOv86.onnx
│
├── rknn/
│   ├── alpha-300.rknn
│   ├── mon_YOLOv84.rknn
│   └── Version6.rknn
│
├── runs/
│   └── detect/
│       └── models/
│           ├── mon_YOLOv8/
│           ├── mon_YOLOv82/
│           ├── mon_YOLOv83/
│           ├── mon_YOLOv84/
│           ├── mon_YOLOv85/
│           └── mon_YOLOv86/
│
├── scripts/
│   ├── convert_npy_to_png.py
│   ├── convert_to_onnx.py
│   ├── convert_to_rknn.py
│   └── train.py
│
└── server/
    └── backend_ws_rknn.py
```

---

## Description des principaux fichiers

### `main.py`

Script principal pour la ROCK 5B.

- Charge un modèle RKNN depuis `rknn/` (`alpha-300.rknn`, `mon_YOLOv84.rknn`, `Version6.rknn`, etc.).
- Initialise la caméra connectée à la ROCK 5B.
- Effectue le pré‑traitement des frames (redimensionnement, normalisation).
- Lance l’inférence sur le NPU via le runtime RKNN.
- Affiche ou renvoie les détections (boîtes, labels, scores).

### `mainPC.py`

Script d’inférence sur PC.

- Charge un modèle YOLOv8 au format PyTorch (`.pt`) ou ONNX (`.onnx`).
- Lit un flux caméra ou des fichiers vidéo/images.
- Applique le pipeline de pré/post‑traitement pour la détection de personnes.

### `requirements.txt`

Liste les dépendances Python du projet (YOLOv8 / ultralytics, NumPy, OpenCV, RKNN toolkit Python, etc.).

---

## Dossiers de données et de modèles

### `dataset/` et `dataset/data.yaml`

- `dataset/data.yaml` : configuration YOLOv8 du jeu de données.

Exemple minimal :

```yaml
path: ./dataset        # racine du dataset
train: images/train    # images d'entraînement
val: images/val        # images de validation

names:
  0: person
```

Adapte les chemins à ton organisation réelle (dossier images, labels, etc.).

### `dataset_npy/`

- Contient les frames brutes enregistrées au format NumPy (`frame_*.npy`).
- Chaque fichier représente une frame thermique (matrice 32×24 ou dérivée) issue du capteur MLX90640.
- Ces fichiers sont convertis en images exploitables pour YOLOv8 via `scripts/convert_npy_to_png.py`.

### `onnx/`

- Contient les modèles exportés au format ONNX :
  - `best.onnx`
  - `mon_YOLOv84.onnx`
  - `mon_YOLOv86.onnx`
- Utilisés soit pour l’inférence sur PC, soit comme entrée de la conversion RKNN.

### `rknn/`

- Contient les modèles compilés pour la ROCK 5B :
  - `alpha-300.rknn`
  - `mon_YOLOv84.rknn`
  - `Version6.rknn`
- Chargés par `main.py` et `backend_ws_rknn.py` pour l’inférence sur le NPU.

### `runs/detect/models/mon_YOLOv8*`

Résultats d’entraînement YOLOv8 (plusieurs expérimentations) :

- `args.yaml` : hyperparamètres et configuration d’entraînement.
- `results.csv`, `results.png` : métriques par epoch / courbes de performance.
- `BoxF1_curve.png`, `BoxPR_curve.png`, `BoxP_curve.png`, `BoxR_curve.png` : courbes F1, précision, rappel.
- `confusion_matrix*.png` : matrices de confusion (brute et normalisée).
- Divers `.jpg` : batchs d’entraînement/validation avec labels/prédictions.
- Sous‑dossier `weights/` :
  - `best.pt` : meilleur modèle.
  - `last.pt` : modèle de la dernière epoch.

---

## Scripts d’entraînement et de conversion

### `scripts/train.py`

- Lance l’entraînement YOLOv8 en utilisant `dataset/data.yaml`.
- Enregistre les résultats d’entraînement dans `runs/detect/models/mon_YOLOv8*`.

Exemple :

```bash
python scripts/train.py --data dataset/data.yaml --name mon_YOLOv8 --epochs 100
```

### `scripts/convert_npy_to_png.py`

- Parcourt `dataset_npy/` et charge chaque `frame_*.npy`.
- Convertit les matrices NumPy en images (niveau de gris ou colormap).
- Sauvegarde les images dans un dossier dédié (par ex. `dataset/images/`) pour l’annotation et l’entraînement.

Exemple :

```bash
python scripts/convert_npy_to_png.py --input dataset_npy --output dataset/images
```

### `scripts/convert_to_onnx.py`

- Charge un modèle YOLOv8 (`.pt`) depuis `runs/detect/models/.../weights/`.
- Exporte le modèle au format ONNX vers le dossier `onnx/`.

Exemple :

```bash
python scripts/convert_to_onnx.py \
  --weights runs/detect/models/mon_YOLOv84/weights/best.pt \
  --output onnx/mon_YOLOv84.onnx
```

### `scripts/convert_to_rknn.py`

- Charge un modèle ONNX depuis `onnx/`.
- Convertit le modèle en RKNN (quantification, optimisation pour NPU).
- Sauvegarde le modèle RKNN dans `rknn/`.

Exemple :

```bash
python scripts/convert_to_rknn.py \
  --onnx onnx/mon_YOLOv84.onnx \
  --output rknn/mon_YOLOv84.rknn
```

---

## Backend serveur

### `server/backend_ws_rknn.py`

- Charge un modèle RKNN depuis `rknn/`.
- Démarre un serveur (WebSocket/HTTP) acceptant des images/frames en entrée.
- Retourne les détections (boîtes, labels, scores) aux clients distants.

Utile pour dissocier :
- le calcul (ROCK 5B + NPU),
- de l’interface (PC, application web, etc.).

---

## Connexion caméra MLX90640 et modèle

Ce projet suppose l’utilisation d’une caméra thermique basée sur le capteur MLX90640 (matrice 32×24, capteur IR de mesure de température).[web:14][web:54]  

La partie **acquisition des données** (câblage de la caméra, configuration I²C, récupération des 32×24 valeurs de température, etc.) peut être réalisée en s’appuyant sur le projet suivant :

- Projet MLX90640 : https://github.com/gaesty/MLX90640BAA  

### Rôle de chaque projet

1. **Projet MLX90640BAA**
   - Gère le matériel (MLX90640 + microcontrôleur / carte hôte).
   - Configure le capteur (I²C, fréquence de rafraîchissement, etc.).
   - Lit les 32×24 valeurs de température et les transforme en :
     - matrices NumPy (`.npy`) ;
     - ou images (PNG/JPEG) générées à partir de ces matrices.
   - Fournit ainsi les frames brutes pour ce projet.

2. **Ce projet (`Human_detection_rock5b`)**
   - Récupère les frames produites par le projet MLX90640BAA (fichiers `.npy` dans `dataset_npy/` ou images converties dans `dataset/images/`).
   - Convertit les `.npy` en images annotables via `scripts/convert_npy_to_png.py`.
   - Entraîne un modèle YOLOv8 sur ces images (`scripts/train.py`).
   - Exporte le modèle vers ONNX puis RKNN.
   - Déploie le modèle pour la détection de personnes :
     - sur PC (`mainPC.py`) ;
     - sur ROCK 5B (`main.py` ou `server/backend_ws_rknn.py`).

En résumé : **MLX90640BAA** gère la **connexion et la lecture de la caméra**, tandis que **Human_detection_rock5b** gère la **détection de personnes à partir des images/frames produites**.

---

## Pipeline global

1. **Acquisition / collecte**
   - Utiliser le projet MLX90640BAA pour lire le capteur MLX90640 et enregistrer les frames (en `.npy` ou en images).

2. **Préparation des données**
   - Placer les `.npy` dans `dataset_npy/`.
   - Convertir en images avec `scripts/convert_npy_to_png.py`.
   - Annoter les images et mettre à jour `dataset/data.yaml`.

3. **Entraînement YOLOv8**
   - Lancer `scripts/train.py` pour entraîner plusieurs variantes `mon_YOLOv8*`.
   - Vérifier les métriques et choisir le meilleur modèle (`best.pt`).

4. **Export du modèle**
   - Exporter le modèle entraîné en ONNX (`scripts/convert_to_onnx.py`).
   - Convertir le modèle ONNX en RKNN (`scripts/convert_to_rknn.py`).

5. **Déploiement / inférence**
   - Sur PC : `mainPC.py` avec modèle `.pt` ou `.onnx`.
   - Sur ROCK 5B :
     - `main.py` pour l’inférence locale,
     - ou `server/backend_ws_rknn.py` pour une API réseau.

---

## Captures d’écran

Les fichiers `Screenshot_*.png` et `screen_entrainement.png` illustrent :
- l’entraînement YOLOv8 (courbes de performance, matrices de confusion, etc.) ;
- des exemples de détection sur des images ou flux vidéo.

