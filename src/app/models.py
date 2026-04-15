"""Database models for the Human Detection API."""

from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.postgresql import INET
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()
IP_ADDRESS_TYPE = db.String(50).with_variant(INET(), "postgresql")


class User(db.Model):
    """Application user used by Flutter supervisor app."""

    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(120), unique=True, nullable=False, index=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    first_name = db.Column(db.String(120), nullable=False)
    last_name = db.Column(db.String(120), nullable=False)
    profile_image_path = db.Column(db.String(1024), nullable=True)
    is_validated = db.Column(db.Boolean, default=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    esp_nodes = db.relationship("ESPNode", back_populates="user")
    scenarios = db.relationship("Scenario", back_populates="user")
    logs = db.relationship("Logging", back_populates="user")

    def set_password(self, raw_password: str) -> None:
        self.password_hash = generate_password_hash(raw_password)

    def check_password(self, raw_password: str) -> bool:
        return check_password_hash(self.password_hash, raw_password)

    def to_dict(self):
        """Convert to dictionary without sensitive fields."""
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "profile_image_path": self.profile_image_path,
            "is_validated": self.is_validated,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class ESPNode(db.Model):
    """Represents an ESP32 node that sends thermal data."""
    
    __tablename__ = "esp_nodes"
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True, index=True)
    ip_address = db.Column(IP_ADDRESS_TYPE, unique=True, nullable=False, index=True)
    room_name = db.Column(db.String(255), nullable=True)
    camera_url = db.Column(db.String(1024), nullable=True)
    color_hex = db.Column(db.String(16), nullable=True)
    pos_x = db.Column(db.Float, nullable=False, default=50.0)
    pos_y = db.Column(db.Float, nullable=False, default=50.0)
    has_camera = db.Column(db.Boolean, nullable=False, default=True)
    show_temperature = db.Column(db.Boolean, nullable=False, default=True)
    show_presence = db.Column(db.Boolean, nullable=False, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    user = db.relationship("User", back_populates="esp_nodes")
    temperatures = db.relationship("Temperature", back_populates="esp_node", cascade="all, delete-orphan")
    scenarios = db.relationship("Scenario", secondary="scenario_esp_nodes", back_populates="esp_nodes")
    
    def __repr__(self):
        return f"<ESPNode {self.ip_address} - {self.room_name}>"
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "ip_address": self.ip_address,
            "room_name": self.room_name,
            "camera_url": self.camera_url,
            "color_hex": self.color_hex,
            "pos_x": self.pos_x,
            "pos_y": self.pos_y,
            "has_camera": self.has_camera,
            "show_temperature": self.show_temperature,
            "show_presence": self.show_presence,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class Temperature(db.Model):
    """Stores temperature measurements from ESP nodes."""
    
    __tablename__ = "temperatures"
    
    id = db.Column(db.Integer, primary_key=True)
    esp_node_id = db.Column(db.Integer, db.ForeignKey("esp_nodes.id"), nullable=False)
    event_key = db.Column(db.String(255), unique=True, nullable=False, index=True)
    temperature = db.Column(db.Numeric(6, 2), nullable=False)
    measured_at = db.Column(db.DateTime, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    esp_node = db.relationship("ESPNode", back_populates="temperatures")
    
    # Indexes for performance
    __table_args__ = (
        db.Index("ix_temperatures_node_measured", "esp_node_id", "measured_at"),
    )
    
    def __repr__(self):
        return f"<Temperature {self.event_key}: {self.temperature}C at {self.measured_at}>"
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": self.id,
            "esp_node_id": self.esp_node_id,
            "event_key": self.event_key,
            "temperature": float(self.temperature) if self.temperature else None,
            "measured_at": self.measured_at.isoformat() if self.measured_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class Logging(db.Model):
    """Audit log for system and user actions."""
    
    __tablename__ = "logging"
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True, index=True)
    log_type = db.Column(db.String(50), nullable=False, index=True)  # 'user' or 'system'
    action_log = db.Column(db.Text, nullable=False)
    concerned_column = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    user = db.relationship("User", back_populates="logs")
    
    def __repr__(self):
        return f"<Logging {self.log_type}: {self.action_log[:50]}...>"
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "username": self.user.username if self.user else None,
            "log_type": self.log_type,
            "action_log": self.action_log,
            "concerned_column": self.concerned_column,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class Scenario(db.Model):
    """Represents a detection scenario configuration."""
    
    __tablename__ = "scenarios"
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True, index=True)
    name = db.Column(db.String(255), nullable=False, index=True)
    description = db.Column(db.Text, nullable=True)
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    icon_code = db.Column(db.Integer, nullable=True)
    color_value = db.Column(db.BigInteger, nullable=True)
    start_hour = db.Column(db.Integer, nullable=True)
    start_minute = db.Column(db.Integer, nullable=True)
    end_hour = db.Column(db.Integer, nullable=True)
    end_minute = db.Column(db.Integer, nullable=True)
    target_temp = db.Column(db.Numeric(6, 2), nullable=True)
    use_time_limit = db.Column(db.Boolean, nullable=False, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    user = db.relationship("User", back_populates="scenarios")
    esp_nodes = db.relationship("ESPNode", secondary="scenario_esp_nodes", back_populates="scenarios")

    __table_args__ = (
        db.UniqueConstraint("user_id", "name", name="uq_scenarios_user_name"),
    )
    
    def __repr__(self):
        return f"<Scenario {self.name} (active={self.is_active})>"
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "description": self.description,
            "is_active": self.is_active,
            "icon_code": self.icon_code,
            "color_value": self.color_value,
            "start_hour": self.start_hour,
            "start_minute": self.start_minute,
            "end_hour": self.end_hour,
            "end_minute": self.end_minute,
            "target_temp": float(self.target_temp) if self.target_temp is not None else None,
            "use_time_limit": self.use_time_limit,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "esp_nodes": [node.to_dict() for node in self.esp_nodes]
        }


class ScenarioESPNode(db.Model):
    """Junction table for many-to-many relationship between Scenarios and ESPNodes."""
    
    __tablename__ = "scenario_esp_nodes"
    
    scenario_id = db.Column(db.Integer, db.ForeignKey("scenarios.id"), primary_key=True)
    esp_node_id = db.Column(db.Integer, db.ForeignKey("esp_nodes.id"), primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    
    def __repr__(self):
        return f"<ScenarioESPNode scenario_id={self.scenario_id}, esp_node_id={self.esp_node_id}>"
