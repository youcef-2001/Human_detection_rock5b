# Human Detection API (Reference Unique)

Ce document fusionne et remplace les anciennes documentations API.
Il décrit les routes réellement implémentées dans le backend Flask actuel.

## Pré-requis d'exécution

- Python 3.11.x (obligatoire, vérifié au démarrage par `src/run.py`).
- Base PostgreSQL démarrée via `docker compose` dans `src/`.
- Variables d'environnement définies via `src/.env`.

### Démarrage local recommandé

```bash
cd src
cp .env.example .env

# Base principale (port hôte 5432)
docker compose up -d postgres

# Dépendances backend
make install

# API Flask (sur l'hôte)
make run
```

### Exécution des tests

Les tests utilisent la base de test (port hôte 5433):

```bash
cd src
docker compose up -d postgres_test
make test
```

## URL de base

- API: `http://localhost:5000`

## Liste complète des routes

### Système

- GET /health
- GET /hello/

### Inference

- POST /inference/detect

### Auth et utilisateurs

- POST /api/auth/signup
- POST /api/auth/login
- GET /api/users/{username}

### ESP32 nodes

- GET /api/esp-nodes
- GET /api/esp-nodes/{node_id}
- POST /api/esp-nodes
- PUT /api/esp-nodes/{node_id}
- DELETE /api/esp-nodes/{node_id}

### Températures

- GET /api/temperatures
- GET /api/temperatures/{temp_id}
- POST /api/temperatures
- PUT /api/temperatures/{temp_id}
- DELETE /api/temperatures/{temp_id}

### Logs

- GET /api/logging
- GET /api/logging/{log_id}
- POST /api/logging
- PUT /api/logging/{log_id}
- DELETE /api/logging/{log_id}
- GET /api/logging/stats

### Scénarios

- GET /api/scenarios
- GET /api/scenarios/{scenario_id}
- POST /api/scenarios
- PUT /api/scenarios/{scenario_id}
- DELETE /api/scenarios/{scenario_id}
- POST /api/scenarios/{scenario_id}/esp-nodes
- DELETE /api/scenarios/{scenario_id}/esp-nodes/{esp_node_id}

### Découverte réseau

- POST /api/network/scan-esp32

## Endpoints système

### Health Check

```http
GET /health
```

Réponse 200:

```json
{
    "status": "healthy"
}
```

### Hello

```http
GET /hello/
```

Réponse 200:

```json
{
    "message": "Hello World"
}
```

## Inference

### Détection

```http
POST /inference/detect
```

Formats supportés:

- `multipart/form-data` avec champ fichier `image`.
- `application/json` (payload thermique 24x32 décodable en 768 float32).

Réponse 200:

```json
{
    "human_count": 2,
    "hot_object_count": 1,
    "success": true
}
```

Erreurs usuelles:

- `400` payload invalide.
- `503` backend d'inférence indisponible.
- `500` erreur interne.

## Authentification et utilisateurs

### Inscription

```http
POST /api/auth/signup
Content-Type: application/json
```

Corps minimal:

```json
{
    "username": "alice",
    "email": "alice@example.com",
    "password": "secret",
    "first_name": "Alice",
    "last_name": "Doe"
}
```

Erreurs:

- `400` champ obligatoire absent.
- `409` username ou email déjà existant.

### Login

```http
POST /api/auth/login
Content-Type: application/json
```

Corps:

```json
{
    "username": "alice",
    "password": "secret"
}
```

Erreurs:

- `400` payload incomplet.
- `401` identifiants invalides.
- `403` compte non validé.

### Profil utilisateur

```http
GET /api/users/{username}
```

Erreurs:

- `404` utilisateur introuvable.

## ESP32 Nodes

### Lister les noeuds

```http
GET /api/esp-nodes?username=alice
```

Paramètres:

- `username` (optionnel): filtre par propriétaire.

### Obtenir un noeud

```http
GET /api/esp-nodes/{node_id}
```

### Créer un noeud

```http
POST /api/esp-nodes
Content-Type: application/json
```

Corps minimal:

```json
{
    "ip_address": "192.168.1.100"
}
```

Corps complet possible:

```json
{
    "username": "alice",
    "ip_address": "192.168.1.100",
    "room_name": "Salon",
    "camera_url": "http://192.168.1.100",
    "color_hex": "#FF5500",
    "pos_x": 50,
    "pos_y": 40,
    "has_camera": true,
    "show_temperature": true,
    "show_presence": true
}
```

Erreurs:

- `400` `ip_address` manquant.
- `404` username inconnu.
- `409` IP déjà existante.

### Mettre à jour un noeud

```http
PUT /api/esp-nodes/{node_id}
Content-Type: application/json
```

Champs modifiables: `room_name`, `camera_url`, `color_hex`, `pos_x`, `pos_y`, `has_camera`, `show_temperature`, `show_presence`.

### Supprimer un noeud

```http
DELETE /api/esp-nodes/{node_id}
```

## Températures

### Lister

```http
GET /api/temperatures?esp_node_id=1&limit=100&offset=0
```

### Obtenir

```http
GET /api/temperatures/{temp_id}
```

### Créer

```http
POST /api/temperatures
Content-Type: application/json
```

Corps requis:

```json
{
    "esp_node_id": 1,
    "event_key": "sensor_001_1713176400",
    "temperature": 25.5,
    "measured_at": "2024-04-15T10:30:00"
}
```

Erreurs:

- `400` champ manquant ou date invalide (ISO 8601 attendu).
- `404` noeud ESP introuvable.
- `409` `event_key` déjà existant.

### Mettre à jour

```http
PUT /api/temperatures/{temp_id}
Content-Type: application/json
```

Champs modifiables: `temperature`, `measured_at`.

### Supprimer

```http
DELETE /api/temperatures/{temp_id}
```

## Logs

### Lister

```http
GET /api/logging?username=alice&log_type=user&limit=100&offset=0
```

Paramètres:

- `username` (optionnel)
- `log_type` (optionnel: `user` ou `system`)
- `limit`, `offset`

### Obtenir

```http
GET /api/logging/{log_id}
```

### Créer

```http
POST /api/logging
Content-Type: application/json
```

Exemple:

```json
{
    "username": "alice",
    "log_type": "user",
    "action_log": "Scenario created",
    "concerned_column": "scenarios"
}
```

Erreurs:

- `400` `log_type`/`action_log` manquants ou type invalide.
- `404` username inconnu.

### Mettre à jour

```http
PUT /api/logging/{log_id}
Content-Type: application/json
```

Champs modifiables: `action_log`, `log_type`, `concerned_column`.

### Supprimer

```http
DELETE /api/logging/{log_id}
```

### Statistiques

```http
GET /api/logging/stats
```

Réponse 200:

```json
{
    "total": 150,
    "user_logs": 100,
    "system_logs": 50
}
```

## Scénarios

### Lister

```http
GET /api/scenarios?username=alice&is_active=true&limit=100&offset=0
```

### Obtenir

```http
GET /api/scenarios/{scenario_id}
```

### Créer

```http
POST /api/scenarios
Content-Type: application/json
```

Exemple:

```json
{
    "username": "alice",
    "name": "Living Room Detection",
    "description": "Detect humans in living room",
    "is_active": true,
    "icon_code": 58826,
    "color_value": 4283215696,
    "start_hour": 8,
    "start_minute": 0,
    "end_hour": 20,
    "end_minute": 0,
    "target_temp": 35.5,
    "use_time_limit": true,
    "esp_node_ids": [1, 2]
}
```

Erreurs:

- `400` `name` manquant.
- `404` username inconnu.
- `409` nom déjà existant (scope utilisateur).

### Mettre à jour

```http
PUT /api/scenarios/{scenario_id}
Content-Type: application/json
```

Champs modifiables: `description`, `is_active`, `icon_code`, `color_value`, `start_hour`, `start_minute`, `end_hour`, `end_minute`, `target_temp`, `use_time_limit`, `esp_node_ids`.

### Supprimer

```http
DELETE /api/scenarios/{scenario_id}
```

### Ajouter un noeud ESP à un scénario

```http
POST /api/scenarios/{scenario_id}/esp-nodes
Content-Type: application/json
```

Corps:

```json
{
    "esp_node_id": 3
}
```

### Retirer un noeud ESP d'un scénario

```http
DELETE /api/scenarios/{scenario_id}/esp-nodes/{esp_node_id}
```

## Découverte réseau ESP32

### Scanner le réseau

```http
POST /api/network/scan-esp32
Content-Type: application/json
```

Corps optionnel:

```json
{
    "preferred_hosts": ["192.168.1.10", "192.168.1.11"],
    "extra_candidates": ["192.168.1.20"],
    "timeout_ms": 700,
    "max_results": 5,
    "workers": 48,
    "scan_full_subnet": true
}
```

Réponse 200:

```json
{
    "discovered_hosts": ["192.168.1.100"],
    "first_host": "192.168.1.100",
    "scanned_candidate_count": 3,
    "timeout_ms": 700
}
```

## Codes d'erreur HTTP

- `200` succès.
- `201` ressource créée.
- `400` payload invalide.
- `401` non authentifié.
- `403` accès refusé.
- `404` ressource introuvable.
- `409` conflit d'unicité.
- `500` erreur interne.
- `503` service d'inférence indisponible.
