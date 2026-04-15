"""Controllers for ESP32 Node management."""

import logging
from typing import Optional

from flask import Blueprint, request, jsonify
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from ..models import db, ESPNode
from ..services.websocket_service import ESPFleetWebSocketService

esp_nodes_bp = Blueprint("esp_nodes", __name__, url_prefix="/api/esp-nodes")
_esp_fleet_service: Optional[ESPFleetWebSocketService] = None
logger = logging.getLogger(__name__)


def init_esp_fleet_service(service: ESPFleetWebSocketService) -> None:
    """Inject fleet websocket service for scan/registration runtime hooks."""
    global _esp_fleet_service
    _esp_fleet_service = service


@esp_nodes_bp.route("", methods=["GET"])
def list_esp_nodes():
    """
    Get all ESP32 nodes.
    
    Returns:
        JSON list of ESPNode objects.
    """
    try:
        nodes = ESPNode.query.all()
        logger.info("Listed ESP nodes count=%s", len(nodes))
        return jsonify([node.to_dict() for node in nodes]), 200
    except SQLAlchemyError as e:
        return jsonify({"error": f"Database error: {str(e)}"}), 500


@esp_nodes_bp.route("/network-search", methods=["GET"])
def network_search():
    """Scan local network and return available ESP32 websocket hosts."""
    if _esp_fleet_service is None:
        return jsonify({"error": "Fleet service not available"}), 503

    subnet = request.args.get("subnet", default=None, type=str)
    register = request.args.get("register", default="false", type=str).lower() == "true"

    try:
        nodes = _esp_fleet_service.scan_network(subnet_cidr=subnet, register=register)
        logger.info("Network search completed subnet=%s register=%s discovered=%s", subnet, register, len(nodes))
        return jsonify({"count": len(nodes), "nodes": nodes}), 200
    except Exception as e:
        return jsonify({"error": f"Network scan failed: {str(e)}"}), 500


@esp_nodes_bp.route("/register-discovered", methods=["POST"])
def register_discovered():
    """Register discovered IP addresses in DB and start websocket tracking."""
    if _esp_fleet_service is None:
        return jsonify({"error": "Fleet service not available"}), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    ips = data.get("ips", [])
    if not isinstance(ips, list):
        return jsonify({"error": "ips must be a list of IP addresses"}), 400

    try:
        nodes = _esp_fleet_service.register_ips(ips)
        logger.info("Registered discovered ESP nodes requested=%s registered=%s", len(ips), len(nodes))
        return jsonify({"count": len(nodes), "nodes": nodes}), 201
    except SQLAlchemyError as e:
        db.session.rollback()
        return jsonify({"error": f"Database error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Registration failed: {str(e)}"}), 500


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
        logger.info("Fetched ESP node id=%s ip=%s", node.id, node.ip_address)
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
        
        if not ip_address:
            return jsonify({"error": "ip_address is required"}), 400
        
        # Check if node with this IP already exists
        existing = ESPNode.query.filter_by(ip_address=ip_address).first()
        if existing:
            return jsonify({"error": f"Node with IP {ip_address} already exists"}), 409
        
        node = ESPNode(ip_address=ip_address, room_name=room_name)
        db.session.add(node)
        db.session.commit()
        logger.info("Created ESP node id=%s uid=%s ip=%s", node.id, node.node_uid, node.ip_address)

        if _esp_fleet_service is not None:
            _esp_fleet_service.track_node(node.id, node.ip_address)
        
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
        
        db.session.commit()
        logger.info("Updated ESP node id=%s uid=%s room_name=%s", node.id, node.node_uid, node.room_name)
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

        if _esp_fleet_service is not None:
            _esp_fleet_service.untrack_node(node.id)
        
        db.session.delete(node)
        db.session.commit()
        logger.info("Deleted ESP node id=%s uid=%s ip=%s", node.id, node.node_uid, node.ip_address)
        return jsonify({"message": f"Node {node_id} deleted successfully"}), 200
    except SQLAlchemyError as e:
        db.session.rollback()
        return jsonify({"error": f"Database error: {str(e)}"}), 500
