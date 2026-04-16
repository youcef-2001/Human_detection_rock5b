"""Controllers for ESP32 Node management."""

from flask import Blueprint, request, jsonify
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from datetime import datetime

from ..models import db, ESPNode, User

esp_nodes_bp = Blueprint("esp_nodes", __name__, url_prefix="/api/esp-nodes")


@esp_nodes_bp.route("", methods=["GET"])
def list_esp_nodes():
    """
    Get all ESP32 nodes.
    
    Returns:
        JSON list of ESPNode objects.
    """
    try:
        query = ESPNode.query
        username = request.args.get("username", type=str)
        if username:
            user = User.query.filter_by(username=username).first()
            if not user:
                return jsonify([]), 200
            query = query.filter_by(user_id=user.id)

        nodes = query.order_by(ESPNode.created_at.asc()).all()
        return jsonify([node.to_dict() for node in nodes]), 200
    except SQLAlchemyError as e:
        return jsonify({"error": f"Database error: {str(e)}"}), 500


@esp_nodes_bp.route("<int:node_id>", methods=["GET"])
def get_esp_node(node_id):
    """
    Get a specific ESP32 node by ID.
    
    Args:
        node_id: ESP node ID.
    
    Returns:
        JSON ESPNode object or 404.
    """
    try:
        node = ESPNode.query.get(node_id)
        if not node:
            return jsonify({"error": "Node not found"}), 404
        return jsonify(node.to_dict()), 200
    except SQLAlchemyError as e:
        return jsonify({"error": f"Database error: {str(e)}"}), 500


@esp_nodes_bp.route("", methods=["POST"])
def create_esp_node():
    """
    Create a new ESP32 node.
    
    Request JSON:
        {
            "ip_address": "192.168.1.100",
            "room_name": "Living Room"
        }
    
    Returns:
        JSON created ESPNode object or 400/409.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        ip_address = data.get("ip_address")
        room_name = data.get("room_name")
        camera_url = data.get("camera_url")
        color_hex = data.get("color_hex")
        pos_x = data.get("pos_x", 50.0)
        pos_y = data.get("pos_y", 50.0)
        has_camera = bool(data.get("has_camera", True))
        show_temperature = bool(data.get("show_temperature", True))
        show_presence = bool(data.get("show_presence", True))
        username = data.get("username")
        
        if not ip_address:
            return jsonify({"error": "ip_address is required"}), 400
        
        # Check if node with this IP already exists
        existing = ESPNode.query.filter_by(ip_address=ip_address).first()
        if existing:
            return jsonify({"error": f"Node with IP {ip_address} already exists"}), 409
        
        user = None
        if username:
            user = User.query.filter_by(username=username).first()
            if not user:
                return jsonify({"error": f"User '{username}' not found"}), 404

        node = ESPNode(
            user_id=user.id if user else None,
            ip_address=ip_address,
            room_name=room_name,
            camera_url=camera_url,
            color_hex=color_hex,
            pos_x=float(pos_x),
            pos_y=float(pos_y),
            has_camera=has_camera,
            show_temperature=show_temperature,
            show_presence=show_presence,
        )
        db.session.add(node)
        db.session.commit()
        
        return jsonify(node.to_dict()), 201
    except IntegrityError:
        db.session.rollback()
        return jsonify({"error": "IP address already exists"}), 409
    except SQLAlchemyError as e:
        db.session.rollback()
        return jsonify({"error": f"Database error: {str(e)}"}), 500


@esp_nodes_bp.route("<int:node_id>", methods=["PUT"])
def update_esp_node(node_id):
    """
    Update an ESP32 node.
    
    Args:
        node_id: ESP node ID.
    
    Request JSON:
        {
            "room_name": "New Room Name"
        }
    
    Returns:
        JSON updated ESPNode object or 404/400.
    """
    try:
        node = ESPNode.query.get(node_id)
        if not node:
            return jsonify({"error": "Node not found"}), 404
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        if "room_name" in data:
            node.room_name = data["room_name"]

        if "camera_url" in data:
            node.camera_url = data["camera_url"]

        if "color_hex" in data:
            node.color_hex = data["color_hex"]

        if "pos_x" in data:
            node.pos_x = float(data["pos_x"])

        if "pos_y" in data:
            node.pos_y = float(data["pos_y"])

        if "has_camera" in data:
            node.has_camera = bool(data["has_camera"])

        if "show_temperature" in data:
            node.show_temperature = bool(data["show_temperature"])

        if "show_presence" in data:
            node.show_presence = bool(data["show_presence"])
        
        db.session.commit()
        return jsonify(node.to_dict()), 200
    except SQLAlchemyError as e:
        db.session.rollback()
        return jsonify({"error": f"Database error: {str(e)}"}), 500


@esp_nodes_bp.route("<int:node_id>", methods=["DELETE"])
def delete_esp_node(node_id):
    """
    Delete an ESP32 node.
    
    Args:
        node_id: ESP node ID.
    
    Returns:
        JSON success message or 404.
    """
    try:
        node = ESPNode.query.get(node_id)
        if not node:
            return jsonify({"error": "Node not found"}), 404
        
        db.session.delete(node)
        db.session.commit()
        return jsonify({"message": f"Node {node_id} deleted successfully"}), 200
    except SQLAlchemyError as e:
        db.session.rollback()
        return jsonify({"error": f"Database error: {str(e)}"}), 500
