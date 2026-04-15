-- Initial PostgreSQL schema for Human Detection platform.

CREATE TABLE IF NOT EXISTS esp_nodes (
    id BIGSERIAL PRIMARY KEY,
    ip_address INET NOT NULL UNIQUE,
    room_name VARCHAR(128) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS temperatures (
    id BIGSERIAL PRIMARY KEY,
    esp_node_id BIGINT NOT NULL REFERENCES esp_nodes(id) ON DELETE CASCADE,
    event_key VARCHAR(255) NOT NULL UNIQUE,
    temperature NUMERIC(6, 2) NOT NULL,
    measured_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_temperatures_node_time
    ON temperatures (esp_node_id, measured_at DESC);

CREATE TABLE IF NOT EXISTS logging (
    id BIGSERIAL PRIMARY KEY,
    log_type TEXT NOT NULL CHECK (log_type IN ('user', 'system')),
    action_log TEXT NOT NULL,
    concerned_column TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS scenarios (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR(150) NOT NULL UNIQUE,
    description TEXT,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS scenario_esp_nodes (
    scenario_id BIGINT NOT NULL REFERENCES scenarios(id) ON DELETE CASCADE,
    esp_node_id BIGINT NOT NULL REFERENCES esp_nodes(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (scenario_id, esp_node_id)
);
