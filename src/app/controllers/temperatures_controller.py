"""Controllers for Temperature data management."""

import logging
from flask import Blueprint, request, jsonify
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from datetime import datetime

from ..models import db, Temperature, ESPNode

temperatures_bp = Blueprint("temperatures", __name__, url_prefix="/api/temperatures")
logger = logging.getLogger(__name__)


@temperatures_bp.route("", methods=["GET"])
def list_temperatures():
    """
    Get all temperature records.
    
    Query parameters:
        - esp_node_id: Filter by ESP node ID
        - limit: Maximum number of records (default 100)
        - offset: Pagination offset (default 0)
    
    Returns:
        JSON list of Temperature objects.
    """
    try:
        query = Temperature.query
        
        # Filter by ESP node if specified
        esp_node_id = request.args.get("esp_node_id", type=int)
        if esp_node_id:
            query = query.filter_by(esp_node_id=esp_node_id)
        
        # Pagination
        limit = request.args.get("limit", default=100, type=int)
        offset = request.args.get("offset", default=0, type=int)
        
        # Order by measured_at descending (newest first)
        temperatures = query.order_by(Temperature.measured_at.desc()).limit(limit).offset(offset).all()
        logger.info("Listed temperatures count=%s esp_node_id=%s", len(temperatures), esp_node_id)
        
        return jsonify([temp.to_dict() for temp in temperatures]), 200
    except SQLAlchemyError as e:
        return jsonify({"error": f"Database error: {str(e)}"}), 500


@temperatures_bp.route("<int:temp_id>", methods=["GET"])
def get_temperature(temp_id):
    """
    Get a specific temperature record by ID.
    
    Args:
        temp_id: Temperature record ID.
    
    Returns:
        JSON Temperature object or 404.
    """
    try:
        temp = Temperature.query.get(temp_id)
        if not temp:
            return jsonify({"error": "Temperature record not found"}), 404
        logger.info("Fetched temperature id=%s esp_node_id=%s", temp.id, temp.esp_node_id)
        return jsonify(temp.to_dict()), 200
    except SQLAlchemyError as e:
        return jsonify({"error": f"Database error: {str(e)}"}), 500


@temperatures_bp.route("", methods=["POST"])
def create_temperature():
    """
    Create a new temperature record.
    
    Request JSON:
        {
            "esp_node_id": 1,
            "event_key": "sensor_001_timestamp",
            "temperature": 25.5,
            "measured_at": "2024-04-15T10:30:00"
        }
    
    Returns:
        JSON created Temperature object or 400/409.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Validate required fields
        required_fields = ["esp_node_id", "event_key", "temperature", "measured_at"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"{field} is required"}), 400
        
        # Verify ESP node exists
        node = ESPNode.query.get(data["esp_node_id"])
        if not node:
            return jsonify({"error": f"ESP node {data['esp_node_id']} not found"}), 404
        
        # Parse measured_at
        try:
            measured_at = datetime.fromisoformat(data["measured_at"].replace("Z", "+00:00"))
        except ValueError:
            return jsonify({"error": "Invalid measured_at format. Use ISO 8601 format"}), 400
        
        temp = Temperature(
            esp_node_id=data["esp_node_id"],
            event_key=data["event_key"],
            temperature=float(data["temperature"]),
            measured_at=measured_at
        )
        
        db.session.add(temp)
        db.session.commit()
        logger.info("Created temperature id=%s esp_node_id=%s event_key=%s", temp.id, temp.esp_node_id, temp.event_key)
        
        return jsonify(temp.to_dict()), 201
    except IntegrityError:
        db.session.rollback()
        return jsonify({"error": "Event key already exists"}), 409
    except SQLAlchemyError as e:
        db.session.rollback()
        return jsonify({"error": f"Database error: {str(e)}"}), 500
    except ValueError as e:
        return jsonify({"error": f"Invalid data format: {str(e)}"}), 400


@temperatures_bp.route("<int:temp_id>", methods=["PUT"])
def update_temperature(temp_id):
    """
    Update a temperature record.
    
    Args:
        temp_id: Temperature record ID.
    
    Request JSON:
        {
            "temperature": 26.0,
            "measured_at": "2024-04-15T10:30:00"
        }
    
    Returns:
        JSON updated Temperature object or 404/400.
    """
    try:
        temp = Temperature.query.get(temp_id)
        if not temp:
            return jsonify({"error": "Temperature record not found"}), 404
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        if "temperature" in data:
            temp.temperature = float(data["temperature"])
        
        if "measured_at" in data:
            try:
                temp.measured_at = datetime.fromisoformat(data["measured_at"].replace("Z", "+00:00"))
            except ValueError:
                return jsonify({"error": "Invalid measured_at format. Use ISO 8601 format"}), 400
        
        db.session.commit()
        logger.info("Updated temperature id=%s esp_node_id=%s", temp.id, temp.esp_node_id)
        return jsonify(temp.to_dict()), 200
    except SQLAlchemyError as e:
        db.session.rollback()
        return jsonify({"error": f"Database error: {str(e)}"}), 500
    except ValueError as e:
        return jsonify({"error": f"Invalid data format: {str(e)}"}), 400


@temperatures_bp.route("<int:temp_id>", methods=["DELETE"])
def delete_temperature(temp_id):
    """
    Delete a temperature record.
    
    Args:
        temp_id: Temperature record ID.
    
    Returns:
        JSON success message or 404.
    """
    try:
        temp = Temperature.query.get(temp_id)
        if not temp:
            return jsonify({"error": "Temperature record not found"}), 404
        
        db.session.delete(temp)
        db.session.commit()
        logger.info("Deleted temperature id=%s esp_node_id=%s", temp.id, temp.esp_node_id)
        return jsonify({"message": f"Temperature record {temp_id} deleted successfully"}), 200
    except SQLAlchemyError as e:
        db.session.rollback()
        return jsonify({"error": f"Database error: {str(e)}"}), 500
