# API Documentation - Human Detection System

## Overview
The API provides CRUD operations for managing ESP32 nodes, temperature data, audit logs, and detection scenarios.

## Base URL
```
http://localhost:5000/api
```

## Authentication
Currently, no authentication is required. Add authentication in production.

---

## ESP32 Nodes Management

### Get All Nodes
```http
GET /api/esp-nodes
```

**Response (200):**
```json
[
  {
    "id": 1,
    "ip_address": "192.168.1.100",
    "room_name": "Living Room",
    "created_at": "2024-04-15T10:00:00"
  }
]
```

### Get Node by ID
```http
GET /api/esp-nodes/{node_id}
```

**Response (200):**
```json
{
  "id": 1,
  "ip_address": "192.168.1.100",
  "room_name": "Living Room",
  "created_at": "2024-04-15T10:00:00"
}
```

### Create Node
```http
POST /api/esp-nodes
Content-Type: application/json

{
  "ip_address": "192.168.1.100",
  "room_name": "Living Room"
}
```

**Response (201):**
```json
{
  "id": 1,
  "ip_address": "192.168.1.100",
  "room_name": "Living Room",
  "created_at": "2024-04-15T10:00:00"
}
```

**Error Responses:**
- `400`: Missing required fields
- `409`: IP address already exists

### Update Node
```http
PUT /api/esp-nodes/{node_id}
Content-Type: application/json

{
  "room_name": "Bedroom"
}
```

**Response (200):** Updated node object

**Error Responses:**
- `404`: Node not found
- `400`: No JSON data provided

### Delete Node
```http
DELETE /api/esp-nodes/{node_id}
```

**Response (200):**
```json
{
  "message": "Node 1 deleted successfully"
}
```

**Error Responses:**
- `404`: Node not found

---

## Temperature Data

### Get All Temperatures
```http
GET /api/temperatures?esp_node_id=1&limit=100&offset=0
```

**Query Parameters:**
- `esp_node_id` (optional): Filter by ESP node
- `limit` (default: 100): Maximum records
- `offset` (default: 0): Pagination offset

**Response (200):**
```json
[
  {
    "id": 1,
    "esp_node_id": 1,
    "event_key": "sensor_001_1713176400",
    "temperature": 25.5,
    "measured_at": "2024-04-15T10:30:00",
    "created_at": "2024-04-15T10:30:05"
  }
]
```

### Get Temperature by ID
```http
GET /api/temperatures/{temp_id}
```

**Response (200):** Temperature object

### Create Temperature Record
```http
POST /api/temperatures
Content-Type: application/json

{
  "esp_node_id": 1,
  "event_key": "sensor_001_1713176400",
  "temperature": 25.5,
  "measured_at": "2024-04-15T10:30:00"
}
```

**Response (201):** Created temperature object

**Error Responses:**
- `400`: Missing required fields
- `404`: ESP node not found
- `409`: Event key already exists

### Update Temperature
```http
PUT /api/temperatures/{temp_id}
Content-Type: application/json

{
  "temperature": 26.0,
  "measured_at": "2024-04-15T10:31:00"
}
```

**Response (200):** Updated temperature object

### Delete Temperature
```http
DELETE /api/temperatures/{temp_id}
```

**Response (200):**
```json
{
  "message": "Temperature record 1 deleted successfully"
}
```

---

## Audit Logs

### Get All Logs
```http
GET /api/logging?log_type=user&limit=100&offset=0
```

**Query Parameters:**
- `log_type` (optional): 'user' or 'system'
- `limit` (default: 100): Maximum records
- `offset` (default: 0): Pagination offset

**Response (200):**
```json
[
  {
    "id": 1,
    "log_type": "user",
    "action_log": "User logged in",
    "concerned_column": "users",
    "created_at": "2024-04-15T10:00:00"
  }
]
```

### Get Log by ID
```http
GET /api/logging/{log_id}
```

### Create Log Entry
```http
POST /api/logging
Content-Type: application/json

{
  "log_type": "user",
  "action_log": "User created new scenario",
  "concerned_column": "scenarios"
}
```

**Response (201):** Created log object

**Error Responses:**
- `400`: Missing required fields
- `400`: Invalid log_type

### Update Log
```http
PUT /api/logging/{log_id}
Content-Type: application/json

{
  "action_log": "Updated log message"
}
```

### Delete Log
```http
DELETE /api/logging/{log_id}
```

### Get Log Statistics
```http
GET /api/logging/stats
```

**Response (200):**
```json
{
  "total": 150,
  "user_logs": 100,
  "system_logs": 50
}
```

---

## Scenarios

### Get All Scenarios
```http
GET /api/scenarios?is_active=true&limit=100&offset=0
```

**Query Parameters:**
- `is_active` (optional): true/false
- `limit` (default: 100): Maximum records
- `offset` (default: 0): Pagination offset

**Response (200):**
```json
[
  {
    "id": 1,
    "name": "Living Room Detection",
    "description": "Detect humans in living room",
    "is_active": true,
    "created_at": "2024-04-15T10:00:00",
    "esp_nodes": [
      {
        "id": 1,
        "ip_address": "192.168.1.100",
        "room_name": "Living Room",
        "created_at": "2024-04-15T10:00:00"
      }
    ]
  }
]
```

### Get Scenario by ID
```http
GET /api/scenarios/{scenario_id}
```

### Create Scenario
```http
POST /api/scenarios
Content-Type: application/json

{
  "name": "Living Room Detection",
  "description": "Detect humans in living room",
  "is_active": true,
  "esp_node_ids": [1, 2]
}
```

**Response (201):** Created scenario object

**Error Responses:**
- `400`: Name is required
- `409`: Scenario name already exists

### Update Scenario
```http
PUT /api/scenarios/{scenario_id}
Content-Type: application/json

{
  "description": "Updated description",
  "is_active": false,
  "esp_node_ids": [1, 2, 3]
}
```

**Response (200):** Updated scenario object

### Delete Scenario
```http
DELETE /api/scenarios/{scenario_id}
```

**Response (200):**
```json
{
  "message": "Scenario 1 deleted successfully"
}
```

### Add ESP Node to Scenario
```http
POST /api/scenarios/{scenario_id}/esp-nodes
Content-Type: application/json

{
  "esp_node_id": 3
}
```

**Response (200):** Updated scenario object

**Error Responses:**
- `404`: Scenario or ESP node not found

### Remove ESP Node from Scenario
```http
DELETE /api/scenarios/{scenario_id}/esp-nodes/{esp_node_id}
```

**Response (200):** Updated scenario object

---

## Health Check

### Check API Health
```http
GET /api/health
```

**Response (200):**
```json
{
  "status": "healthy"
}
```

---

## Inference

### Run Inference
```http
POST /api/inference/detect
Content-Type: multipart/form-data

image: <binary_image_data>
```

**Response (200):**
```json
{
  "human_count": 2,
  "hot_object_count": 1,
  "success": true
}
```

**Error Responses:**
- `400`: Invalid image format
- `503`: Inference service unavailable

---

## Error Handling

All endpoints return standard HTTP status codes:

- `200`: Success
- `201`: Created
- `400`: Bad Request (validation error)
- `404`: Not Found
- `409`: Conflict (e.g., duplicate entry)
- `500`: Internal Server Error
- `503`: Service Unavailable

Error responses follow this format:
```json
{
  "error": "Error message description"
}
```

---

## Rate Limiting

Not implemented. Consider adding in production.

---

## CORS

CORS is not configured by default. Add Flask-CORS extension if needed:

```python
from flask_cors import CORS
CORS(app)
```

---

## Testing

Use curl, Postman, or any HTTP client to test the endpoints.

Example with curl:
```bash
# Create a node
curl -X POST http://localhost:5000/api/esp-nodes \
  -H "Content-Type: application/json" \
  -d '{"ip_address": "192.168.1.100", "room_name": "Living Room"}'

# Get all nodes
curl http://localhost:5000/api/esp-nodes

# Create a temperature record
curl -X POST http://localhost:5000/api/temperatures \
  -H "Content-Type: application/json" \
  -d '{
    "esp_node_id": 1,
    "event_key": "sensor_001_time",
    "temperature": 25.5,
    "measured_at": "2024-04-15T10:30:00"
  }'
```

---

## Environment Configuration

All configuration is loaded from `.env` file in the `src/` directory:

```env
FLASK_ENV=development
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
SECRET_KEY=your-secret-key
DATABASE_URL=postgresql://user:pass@localhost:5432/db
ESP32_WS_URI=ws://10.105.139.24:81/
```

See `.env.example` for all available configuration options.
