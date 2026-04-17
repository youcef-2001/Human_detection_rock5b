# Human Detection - ROCK 5B (RK3588)

Documentation technique du projet de détection humaine thermique, structuré autour de deux volets complémentaires :

1. Pipeline IA (préparation des données, entraînement, conversion ONNX/RKNN)
2. Backend Flask (API métier, persistance PostgreSQL, découverte réseau des noeuds ESP32)

Ce projet est réalisé dans le cadre du projet de fin d'année (Master 2 IoT et Architecture Logicielle).

## Objectifs

- Construire des modèles de détection à partir de données thermiques.
- Déployer les modèles sur plateforme RK3588 via RKNN (NPU).
- Exposer les fonctions métier et de supervision via une API Flask.
- Persister les données applicatives (noeuds ESP32, scénarios, températures, logs) dans PostgreSQL.

## Vue d'ensemble de l'architecture

Le dépôt combine :

- Une chaîne IA orientée entraînement et conversion de modèles.
- Un backend Python/Flask exécuté sur l'hôte.
- Une base PostgreSQL exécutée en conteneur Docker.

Choix d'architecture recommandé dans ce projet :

- API Flask lancée sur l'hôte pour faciliter l'accès au NPU et aux ressources matérielles.
- PostgreSQL en conteneur pour une installation reproductible, simple à démarrer et isolée.

## Structure du dépôt

### Pipeline IA

- dataset/ : datasets images et labels pour l'entraînement des modèles.
- dataset_npy/ : captures thermiques brutes (format .npy) issues de la caméra MLX.
- scripts/ : scripts de préparation de données, entraînement, conversion et tests d'inférence.
- runs/ : sorties d'entraînement et de fine-tuning.
- onnx/ : modèles convertis vers ONNX (depuis PyTorch).
- rknn/ : modèles convertis vers RKNN (depuis ONNX), prêts pour RK3588.

### Backend API

- src/app/ : coeur de l'application Flask (controllers, services, modèles ORM, configuration).
- src/tests/ : tests unitaires et d'intégration.
- src/db/ : scripts d'initialisation SQL et répertoires de données PostgreSQL.
- src/docker-compose.yml : services PostgreSQL (dev et test).
- src/Makefile : commandes de développement (install, run, test, lint, format).

## Chaîne de traitement IA

Principe de bout en bout :

1. Les données thermiques .npy sont converties en images exploitables pour l'entraînement.
2. Un entraînement produit un checkpoint best.pt dans un dossier runs/.../weights/.
3. Chaque best.pt est converti en ONNX.
4. Le modèle ONNX est converti en RKNN pour exécution sur RK3588.

Références utiles :

- Projet caméra thermique MLX (génération/stream des données) : https://github.com/gaesty/MLX90640BAA
- Installation de l'environnement RKNN (Radxa) : https://docs.radxa.com/en/rock5/rock5c/app-development/ai/rknn-install

## Prérequis

- Linux recommandé pour l'environnement RKNN.
- Python 3.11.x.
- Docker + Docker Compose.
- Environnement RKNN correctement installé (toolkit, runtime, dépendances Radxa).

## Démarrage rapide - Partie IA

Depuis la racine du projet :

```bash
conda activate rknn
pip install -r requirements.txt

# Conversion des frames thermiques en images d'entraînement
python scripts/convert_npy_to_png.py --input dataset_npy --output dataset/images

# Entraînement
python scripts/train.py 

# Conversion PyTorch -> ONNX
python scripts/convert_to_onnx.py 

# Conversion ONNX -> RKNN
python scripts/convert_to_rknn.py \
  --onnx onnx/mon_YOLOv84.onnx \
  --output rknn/mon_YOLOv84.rknn
```

Validation finale possible via les scripts de test d'inférence :

- scripts/test_inference_PC.py (environnement PC)
- scripts/test_inference_rock5b.py (plateforme RK3588)

## Démarrage rapide - Backend Flask + PostgreSQL

### 1) Préparer l'environnement backend

```bash
cd src
cp .env.example .env
```

Vérifier que DATABASE_URL pointe vers PostgreSQL local (conteneur mappé sur 5432), par exemple :

```dotenv
DATABASE_URL=postgresql://human_user:human_pass@localhost:5432/human_detection
```

### 2) Démarrer PostgreSQL via Docker

```bash
docker compose up -d postgres
```

### 3) Installer les dépendances backend

```bash
make install
```

### 4) Lancer l'API Flask sur l'hôte

```bash
make run
```

Important : l'exécution de l'API sur l'hôte est préférée pour l'accès NPU/hardware. Le mode API en conteneur n'est pas le mode de référence du projet.

### 5) Lancer les tests

```bash
docker compose up -d postgres_test
make test
```

## Lancement complet avec l'application superviseur_app

Une fois PostgreSQL et l'API démarrés, lancer l'interface Flutter:

```bash
cd ../superviseur_app
flutter run
```

Ports usuels :

- API Flask : http://localhost:5000
- PostgreSQL : localhost:5432
- Flutter Web/Desktop : port attribué par Flutter (variable selon la cible)

## Réseau et découverte ESP32

Pour l'appairage automatique avec les noeuds ESP32 compatibles :

- La machine hôte (ROCK 5B ou machine d'exécution) doit être sur le même réseau local que les ESP32.
- Le backend effectue une détection des noeuds en scannant notamment le port 81 (signature ESP32 thermique/websocket).
- Les noeuds ESP32 doivent être flashés/configurés avec le projet MLX90640BAA puis connectés au Wi-Fi cible.

## Commandes utiles (backend)

Depuis src/ :

- make install : installer les dépendances Python.
- make run : démarrer l'API en mode debug.
- make run-prod : démarrer l'API en mode production.
- make test : exécuter la suite de tests.
- make test-cov : exécuter les tests avec couverture.
- make lint : vérifier le style.
- make format : formater le code.

## Documentation complémentaire

- Documentation API générale : src/API_README.md

## Notes d'exploitation

- La persistance PostgreSQL est gérée via les volumes Docker déclarés dans src/docker-compose.yml et les répertoires sous src/db/data/.
- Les variables sensibles doivent rester dans src/.env (non versionné).
- Le fichier src/.env.example sert de modèle de configuration.

## Dépannage rapide

- Port 5432 occupé : modifier le mapping de ports dans src/docker-compose.yml.
- PostgreSQL indisponible : vérifier l'état du conteneur (docker ps, docker logs).
- API non démarrée : vérifier que l'environnement Python actif est bien en 3.11.x, puis contrôler src/.env et les dépendances.
- Tests en échec avec `Connection refused` sur `localhost:5433` : démarrer `postgres_test` puis relancer `make test`.
- Découverte ESP32 vide : vérifier le même sous-réseau, la connectivité Wi-Fi et la disponibilité du port 81 sur les noeuds.
