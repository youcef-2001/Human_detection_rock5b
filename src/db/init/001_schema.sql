-- Initial PostgreSQL schema for Human Detection platform.

CREATE TABLE IF NOT EXISTS users (
    id BIGSERIAL PRIMARY KEY,
    username VARCHAR(120) NOT NULL UNIQUE,
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(120) NOT NULL,
    last_name VARCHAR(120) NOT NULL,
    profile_image_path VARCHAR(1024),
    is_validated BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

CREATE TABLE IF NOT EXISTS esp_nodes (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT REFERENCES users(id),
    ip_address INET NOT NULL UNIQUE,
    room_name VARCHAR(255),
    camera_url VARCHAR(1024),
    color_hex VARCHAR(16),
    pos_x FLOAT NOT NULL DEFAULT 50.0,
    pos_y FLOAT NOT NULL DEFAULT 50.0,
    has_camera BOOLEAN NOT NULL DEFAULT TRUE,
    show_temperature BOOLEAN NOT NULL DEFAULT TRUE,
    show_presence BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_esp_nodes_user_id ON esp_nodes(user_id);

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
    user_id BIGINT REFERENCES users(id),
    log_type VARCHAR(50) NOT NULL,
    action_log TEXT NOT NULL,
    concerned_column VARCHAR(255),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_logging_user_id ON logging(user_id);
CREATE INDEX IF NOT EXISTS idx_logging_log_type ON logging(log_type);

CREATE TABLE IF NOT EXISTS scenarios (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT REFERENCES users(id),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    icon_code INTEGER,
    color_value BIGINT,
    start_hour INTEGER,
    start_minute INTEGER,
    end_hour INTEGER,
    end_minute INTEGER,
    target_temp NUMERIC(6, 2),
    use_time_limit BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_scenarios_user_id ON scenarios(user_id);

CREATE TABLE IF NOT EXISTS scenario_esp_nodes (
    scenario_id BIGINT NOT NULL REFERENCES scenarios(id) ON DELETE CASCADE,
    esp_node_id BIGINT NOT NULL REFERENCES esp_nodes(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (scenario_id, esp_node_id)
);
