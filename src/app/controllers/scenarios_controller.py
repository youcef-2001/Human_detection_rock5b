"""Controllers for Scenario management."""

from flask import Blueprint, request, jsonify
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from ..models import db, Scenario, ESPNode, ScenarioESPNode

scenarios_bp = Blueprint("scenarios", __name__, url_prefix="/api/scenarios")


@scenarios_bp.route("", methods=["GET"])
def list_scenarios():
    """
    Get all scenarios.
    
    Query parameters:
        - is_active: Filter by active status (true/false)
        - limit: Maximum number of records (default 100)
        - offset: Pagination offset (default 0)
    
    Returns:
        JSON list of Scenario objects.
    """
    try:
        query = Scenario.query
        
        # Filter by active status if specified
        is_active = request.args.get("is_active", type=lambda x: x.lower() == "true")
        if "is_active" in request.args:
            query = query.filter_by(is_active=is_active)
        
        # Pagination
        limit = request.args.get("limit", default=100, type=int)
        offset = request.args.get("offset", default=0, type=int)
        
        # Order by name
        scenarios = query.order_by(Scenario.name).limit(limit).offset(offset).all()
        
        return jsonify([scenario.to_dict() for scenario in scenarios]), 200
    except SQLAlchemyError as e:
        return jsonify({"error": f"Database error: {str(e)}"}), 500


@scenarios_bp.route("<int:scenario_id>", methods=["GET"])
def get_scenario(scenario_id):
    """
    Get a specific scenario by ID.
    
    Args:
        scenario_id: Scenario ID.
    
    Returns:
        JSON Scenario object or 404.
    """
    try:
        scenario = Scenario.query.get(scenario_id)
        if not scenario:
            return jsonify({"error": "Scenario not found"}), 404
        return jsonify(scenario.to_dict()), 200
    except SQLAlchemyError as e:
        return jsonify({"error": f"Database error: {str(e)}"}), 500


@scenarios_bp.route("", methods=["POST"])
def create_scenario():
    """
    Create a new scenario.
    
    Request JSON:
        {
            "name": "Living Room Detection",
            "description": "Detect humans in living room",
            "is_active": true,
            "esp_node_ids": [1, 2]  # optional
        }
    
    Returns:
        JSON created Scenario object or 400/409.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        name = data.get("name")
        if not name:
            return jsonify({"error": "name is required"}), 400
        
        # Check if scenario with this name already exists
        existing = Scenario.query.filter_by(name=name).first()
        if existing:
            return jsonify({"error": f"Scenario with name '{name}' already exists"}), 409
        
        scenario = Scenario(
            name=name,
            description=data.get("description"),
            is_active=data.get("is_active", True)
        )
        
        # Add ESP nodes if specified
        esp_node_ids = data.get("esp_node_ids", [])
        for node_id in esp_node_ids:
            node = ESPNode.query.get(node_id)
            if node:
                scenario.esp_nodes.append(node)
        
        db.session.add(scenario)
        db.session.commit()
        
        return jsonify(scenario.to_dict()), 201
    except IntegrityError:
        db.session.rollback()
        return jsonify({"error": "Scenario name already exists"}), 409
    except SQLAlchemyError as e:
        db.session.rollback()
        return jsonify({"error": f"Database error: {str(e)}"}), 500


@scenarios_bp.route("<int:scenario_id>", methods=["PUT"])
def update_scenario(scenario_id):
    """
    Update a scenario.
    
    Args:
        scenario_id: Scenario ID.
    
    Request JSON:
        {
            "description": "Updated description",
            "is_active": false,
            "esp_node_ids": [1, 2, 3]  # Replace all associated nodes
        }
    
    Returns:
        JSON updated Scenario object or 404/400.
    """
    try:
        scenario = Scenario.query.get(scenario_id)
        if not scenario:
            return jsonify({"error": "Scenario not found"}), 404
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        if "description" in data:
            scenario.description = data["description"]
        
        if "is_active" in data:
            scenario.is_active = bool(data["is_active"])
        
        # Update ESP nodes if specified
        if "esp_node_ids" in data:
            # Clear existing nodes
            scenario.esp_nodes.clear()
            
            # Add new nodes
            for node_id in data["esp_node_ids"]:
                node = ESPNode.query.get(node_id)
                if node:
                    scenario.esp_nodes.append(node)
        
        db.session.commit()
        return jsonify(scenario.to_dict()), 200
    except SQLAlchemyError as e:
        db.session.rollback()
        return jsonify({"error": f"Database error: {str(e)}"}), 500


@scenarios_bp.route("<int:scenario_id>", methods=["DELETE"])
def delete_scenario(scenario_id):
    """
    Delete a scenario.
    
    Args:
        scenario_id: Scenario ID.
    
    Returns:
        JSON success message or 404.
    """
    try:
        scenario = Scenario.query.get(scenario_id)
        if not scenario:
            return jsonify({"error": "Scenario not found"}), 404
        
        db.session.delete(scenario)
        db.session.commit()
        return jsonify({"message": f"Scenario {scenario_id} deleted successfully"}), 200
    except SQLAlchemyError as e:
        db.session.rollback()
        return jsonify({"error": f"Database error: {str(e)}"}), 500


@scenarios_bp.route("<int:scenario_id>/esp-nodes", methods=["POST"])
def add_esp_node_to_scenario(scenario_id):
    """
    Add an ESP node to a scenario.
    
    Args:
        scenario_id: Scenario ID.
    
    Request JSON:
        {
            "esp_node_id": 1
        }
    
    Returns:
        JSON updated Scenario object or 404/400.
    """
    try:
        scenario = Scenario.query.get(scenario_id)
        if not scenario:
            return jsonify({"error": "Scenario not found"}), 404
        
        data = request.get_json()
        if not data or "esp_node_id" not in data:
            return jsonify({"error": "esp_node_id is required"}), 400
        
        node = ESPNode.query.get(data["esp_node_id"])
        if not node:
            return jsonify({"error": "ESP node not found"}), 404
        
        if node not in scenario.esp_nodes:
            scenario.esp_nodes.append(node)
            db.session.commit()
        
        return jsonify(scenario.to_dict()), 200
    except SQLAlchemyError as e:
        db.session.rollback()
        return jsonify({"error": f"Database error: {str(e)}"}), 500


@scenarios_bp.route("<int:scenario_id>/esp-nodes/<int:esp_node_id>", methods=["DELETE"])
def remove_esp_node_from_scenario(scenario_id, esp_node_id):
    """
    Remove an ESP node from a scenario.
    
    Args:
        scenario_id: Scenario ID.
        esp_node_id: ESP Node ID.
    
    Returns:
        JSON updated Scenario object or 404.
    """
    try:
        scenario = Scenario.query.get(scenario_id)
        if not scenario:
            return jsonify({"error": "Scenario not found"}), 404
        
        node = ESPNode.query.get(esp_node_id)
        if not node:
            return jsonify({"error": "ESP node not found"}), 404
        
        if node in scenario.esp_nodes:
            scenario.esp_nodes.remove(node)
            db.session.commit()
        
        return jsonify(scenario.to_dict()), 200
    except SQLAlchemyError as e:
        db.session.rollback()
        return jsonify({"error": f"Database error: {str(e)}"}), 500
