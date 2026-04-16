-- Drop tables in correct order
DROP TABLE IF EXISTS scenario_esp_nodes CASCADE;
DROP TABLE IF EXISTS temperatures CASCADE;
DROP TABLE IF EXISTS scenarios CASCADE;
DROP TABLE IF EXISTS logging CASCADE;
DROP TABLE IF EXISTS esp_nodes CASCADE;
DROP TABLE IF EXISTS users CASCADE;

-- Recreate with correct schema
CREATE TABLE users (
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

CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email);

CREATE TABLE esp_nodes (
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

CREATE INDEX idx_esp_nodes_user_id ON esp_nodes(user_id);

CREATE TABLE temperatures (
    id BIGSERIAL PRIMARY KEY,
    esp_node_id BIGINT NOT NULL REFERENCES esp_nodes(id) ON DELETE CASCADE,
    event_key VARCHAR(255) NOT NULL UNIQUE,
    temperature NUMERIC(6, 2) NOT NULL,
    measured_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_temperatures_node_time ON temperatures (esp_node_id, measured_at DESC);

CREATE TABLE logging (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT REFERENCES users(id),
    log_type VARCHAR(50) NOT NULL,
    action_log TEXT NOT NULL,
    concerned_column VARCHAR(255),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_logging_user_id ON logging(user_id);
CREATE INDEX idx_logging_log_type ON logging(log_type);

CREATE TABLE scenarios (
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
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_scenarios_user_name UNIQUE (user_id, name)
);

CREATE TABLE scenario_esp_nodes (
    scenario_id BIGINT NOT NULL REFERENCES scenarios(id) ON DELETE CASCADE,
    esp_node_id BIGINT NOT NULL REFERENCES esp_nodes(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (scenario_id, esp_node_id)
);
