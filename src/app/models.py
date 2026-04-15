"""Database models for the Human Detection API."""

from datetime import datetime
import uuid
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.postgresql import INET

db = SQLAlchemy()

# Keep portability for tests (SQLite) while binding as INET on PostgreSQL.
IP_ADDRESS_TYPE = db.String(50).with_variant(INET(), "postgresql")


class ESPNode(db.Model):
    """Represents an ESP32 node that sends thermal data."""
    
    __tablename__ = "esp_nodes"
    
    id = db.Column(db.Integer, primary_key=True)
    node_uid = db.Column(db.String(64), unique=True, nullable=False, index=True, default=lambda: uuid.uuid4().hex)
    ip_address = db.Column(IP_ADDRESS_TYPE, unique=True, nullable=False, index=True)
    room_name = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    temperatures = db.relationship("Temperature", back_populates="esp_node", cascade="all, delete-orphan")
    scenarios = db.relationship("Scenario", secondary="scenario_esp_nodes", back_populates="esp_nodes")
    
    def __repr__(self):
        return f"<ESPNode {self.ip_address} - {self.room_name}>"
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": self.id,
            "node_uid": self.node_uid,
            "ip_address": self.ip_address,
            "room_name": self.room_name,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
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
        db.Index("ix_temperatures_created_at", "created_at"),
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
    log_type = db.Column(db.String(50), nullable=False, index=True)  # 'user' or 'system'
    action_log = db.Column(db.Text, nullable=False)
    concerned_column = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    def __repr__(self):
        return f"<Logging {self.log_type}: {self.action_log[:50]}...>"
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": self.id,
            "log_type": self.log_type,
            "action_log": self.action_log,
            "concerned_column": self.concerned_column,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class Scenario(db.Model):
    """Represents a detection scenario configuration."""
    
    __tablename__ = "scenarios"
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), unique=True, nullable=False, index=True)
    description = db.Column(db.Text, nullable=True)
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    esp_nodes = db.relationship("ESPNode", secondary="scenario_esp_nodes", back_populates="scenarios")
    
    def __repr__(self):
        return f"<Scenario {self.name} (active={self.is_active})>"
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "esp_nodes": [node.to_dict() for node in getattr(self, "esp_nodes", [])]
        }


class ScenarioESPNode(db.Model):
    """Junction table for many-to-many relationship between Scenarios and ESPNodes."""
    
    __tablename__ = "scenario_esp_nodes"
    
    scenario_id = db.Column(db.Integer, db.ForeignKey("scenarios.id"), primary_key=True)
    esp_node_id = db.Column(db.Integer, db.ForeignKey("esp_nodes.id"), primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    
    def __repr__(self):
        return f"<ScenarioESPNode scenario_id={self.scenario_id}, esp_node_id={self.esp_node_id}>"
