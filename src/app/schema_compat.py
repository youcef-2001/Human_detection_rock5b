"""PostgreSQL schema compatibility helpers.

This module patches older databases in place so new ORM fields do not crash
CRUD routes when a persisted docker volume contains an outdated schema.
"""

from sqlalchemy import text


def ensure_postgres_schema_compat(db) -> None:
    """Apply idempotent PostgreSQL ALTER statements for missing columns."""
    engine = db.engine
    if engine.dialect.name != "postgresql":
        return

    statements = [
        # esp_nodes
        "ALTER TABLE IF EXISTS esp_nodes ADD COLUMN IF NOT EXISTS user_id BIGINT",
        "ALTER TABLE IF EXISTS esp_nodes ADD COLUMN IF NOT EXISTS camera_url VARCHAR(1024)",
        "ALTER TABLE IF EXISTS esp_nodes ADD COLUMN IF NOT EXISTS color_hex VARCHAR(16)",
        "ALTER TABLE IF EXISTS esp_nodes ADD COLUMN IF NOT EXISTS pos_x DOUBLE PRECISION NOT NULL DEFAULT 50.0",
        "ALTER TABLE IF EXISTS esp_nodes ADD COLUMN IF NOT EXISTS pos_y DOUBLE PRECISION NOT NULL DEFAULT 50.0",
        "ALTER TABLE IF EXISTS esp_nodes ADD COLUMN IF NOT EXISTS has_camera BOOLEAN NOT NULL DEFAULT TRUE",
        "ALTER TABLE IF EXISTS esp_nodes ADD COLUMN IF NOT EXISTS show_temperature BOOLEAN NOT NULL DEFAULT TRUE",
        "ALTER TABLE IF EXISTS esp_nodes ADD COLUMN IF NOT EXISTS show_presence BOOLEAN NOT NULL DEFAULT TRUE",
        "CREATE INDEX IF NOT EXISTS idx_esp_nodes_user_id ON esp_nodes(user_id)",
        # scenarios
        "ALTER TABLE IF EXISTS scenarios ADD COLUMN IF NOT EXISTS user_id BIGINT",
        "ALTER TABLE IF EXISTS scenarios ADD COLUMN IF NOT EXISTS icon_code INTEGER",
        "ALTER TABLE IF EXISTS scenarios ADD COLUMN IF NOT EXISTS color_value BIGINT",
        "ALTER TABLE IF EXISTS scenarios ADD COLUMN IF NOT EXISTS start_hour INTEGER",
        "ALTER TABLE IF EXISTS scenarios ADD COLUMN IF NOT EXISTS start_minute INTEGER",
        "ALTER TABLE IF EXISTS scenarios ADD COLUMN IF NOT EXISTS end_hour INTEGER",
        "ALTER TABLE IF EXISTS scenarios ADD COLUMN IF NOT EXISTS end_minute INTEGER",
        "ALTER TABLE IF EXISTS scenarios ADD COLUMN IF NOT EXISTS target_temp NUMERIC(6, 2)",
        "ALTER TABLE IF EXISTS scenarios ADD COLUMN IF NOT EXISTS use_time_limit BOOLEAN NOT NULL DEFAULT TRUE",
        "CREATE INDEX IF NOT EXISTS idx_scenarios_user_id ON scenarios(user_id)",
        # logging
        "ALTER TABLE IF EXISTS logging ADD COLUMN IF NOT EXISTS user_id BIGINT",
        "CREATE INDEX IF NOT EXISTS idx_logging_user_id ON logging(user_id)",
    ]

    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))
