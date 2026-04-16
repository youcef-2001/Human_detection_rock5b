# Human_detection_rock5b

Ce dépôt contient 2 projets complémentaires :

1. Projet IA RK SDK (entraînement, conversion, optimisation NPU RKNN)
2. Projet serveur web Human Detector (API Flask + PostgreSQL )

Le but est simple :
- préparer un modèle de détection de personnes à partir de données thermiques ;
- l’exécuter et l’exposer via une API utilisable facilement.

---

## Vue rapide

### Projet 1 : IA / RK SDK

À quoi ça sert :
- créer et améliorer les modèles IA ;
- convertir les données thermiques ;
- entraîner YOLO ;
- exporter vers ONNX puis RKNN pour ROCK 5B.

Dossiers clés :
- scripts : scripts de conversion/entraînement
- dataset et dataset_npy : données d’entrée
- onnx : modèles ONNX
- rknn : modèles pour NPU
- runs : résultats d’entraînement

### Projet 2 : Serveur web Human Detector

À quoi ça sert :
- exposer une API HTTP ;
- stocker les ESP nodes, les scenarios, les logs de température pour l'application Front;
- conserver les données en base PostgreSQL même après redémarrage Docker.

Dossiers clés :
- src/app : code de l’API
- src/docker-compose.yml : service PostgreSQL
- src/db/init : scripts SQL d’initialisation
- src/tests : tests API

Persistance DB :
- PostgreSQL écrit ses données sur un volume bindé : /volumes/data/postgresql

---

## Prérequis minimum

- Linux recommandé
- Python 3.11
- pip
- Docker + Docker Compose
- RKNN SDK 

---

## Démarrage rapide (Application superviseur_app)

Pour lancer l'application complète avec le backend Flask et PostgreSQL :

### Terminal 1 - Démarrer PostgreSQL (Docker)

```bash
cd src
docker-compose up -d
```

### Terminal 2 - Démarrer l'API Flask

```bash
cd .
(& ".\.venv\Scripts\Activate.ps1")  # Activation virtualenv Windows
python "src\run.py" --host 0.0.0.0 --port 5000
```

**Ou avec PowerShell complet :**
```powershell
(& "c:\Users\boula\Desktop\Human_detection_rock5b\.venv\Scripts\Activate.ps1") ; python "src\run.py" --host 0.0.0.0 --port 5000
```

### Terminal 3 - Lancer l'application Flutter

```bash
cd ../superviseur_app
flutter run
```

**Résumé des ports :**
- API Flask : `http://localhost:5000`
- PostgreSQL : `localhost:5432`
- Application Flutter : `http://localhost:54107` (ou autre)

**L'API sera accessible à `http://localhost:5000`** et l'application Flutter s'y connectera automatiquement.

---

## Get Started 1 - IA RK SDK (démarrage rapide)

Objectif : entraîner/convertir un modèle.

1. Aller à la racine du projet

```bash
cd Human_detection_rock5b
```

2. Activer l’environnement Python

```bash
conda activate rknn
```

3. Installer les dépendances

```bash
pip install -r requirements.txt
```

4. Convertir les frames .npy en images

```bash
python scripts/convert_npy_to_png.py --input dataset_npy --output dataset/images
```

5. Entraîner YOLO

```bash
python scripts/train.py --data dataset/data.yaml --name mon_YOLOv8 --epochs 100
```

6. Exporter en ONNX

```bash
python scripts/convert_to_onnx.py \
  --weights runs/detect/models/mon_YOLOv84/weights/best.pt \
  --output onnx/mon_YOLOv84.onnx
```

7. Convertir ONNX vers RKNN

```bash
python scripts/convert_to_rknn.py \
  --onnx onnx/mon_YOLOv84.onnx \
  --output rknn/mon_YOLOv84.rknn
```

Résultat attendu :
- un modèle ONNX dans onnx/
- un modèle RKNN dans rknn/

---

## Get Started 2 - Serveur web (démarrage rapide)

Objectif : lancer l’API en local avec PostgreSQL dans Docker.

1. Aller dans le projet serveur

```bash
cd src
```

2. Créer le fichier d’environnement local (si absent)

```bash
cp .env.example .env
```

3. Vérifier la base (mode API en local + DB Docker)

- DATABASE_URL doit pointer sur localhost:5432

Exemple :

```dotenv
DATABASE_URL=postgresql://human_user:human_pass@localhost:5432/human_detection
```

4. Démarrer PostgreSQL uniquement

```bash
docker compose up -d postgres
```

5. Installer les dépendances serveur

```bash
make install
```

6. Lancer l’API sur la machine hôte

```bash
make run
```

7. Tester rapidement

```bash
curl http://localhost:5000/health
```

---

## Variables et sécurité

- src/.env : valeurs réelles locales (secrets, URLs, mots de passe)
- src/.env.example : modèle partageable sur GitHub


---

## Dépannage rapide

- Port 5432 occupé : changer le mapping dans src/docker-compose.yml
- Erreur DB refused : vérifier que postgres est bien up (docker ps)
- API ne démarre pas : vérifier que src/.env existe et est rempli
- Données perdues : vérifier que /volumes/data/postgresql est accessible

---

## En résumé

- Projet IA RK SDK = fabriquer et optimiser les modèles.
- Projet serveur web = exposer les services métiers + stocker les données.
- Pour un usage simple : lancer postgres via Docker, lancer API en local, puis appeler les routes HTTP.
