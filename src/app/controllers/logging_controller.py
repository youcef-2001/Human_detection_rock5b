"""Controllers for Logging/Audit management."""

import logging
from flask import Blueprint, request, jsonify
from sqlalchemy.exc import SQLAlchemyError

from ..models import db, Logging

logging_bp = Blueprint("logging", __name__, url_prefix="/api/logging")
logger = logging.getLogger(__name__)


@logging_bp.route("", methods=["GET"])
def list_logs():
    """
    Get all audit logs.
    
    Query parameters:
        - log_type: Filter by log type ('user' or 'system')
        - limit: Maximum number of records (default 100)
        - offset: Pagination offset (default 0)
    
    Returns:
        JSON list of Logging objects.
    """
    try:
        query = Logging.query
        
        # Filter by log type if specified
        log_type = request.args.get("log_type")
        if log_type and log_type in ["user", "system"]:
            query = query.filter_by(log_type=log_type)
        
        # Pagination
        limit = request.args.get("limit", default=100, type=int)
        offset = request.args.get("offset", default=0, type=int)
        
        # Order by created_at descending (newest first)
        logs = query.order_by(Logging.created_at.desc()).limit(limit).offset(offset).all()
        logger.info("Listed audit logs count=%s log_type=%s", len(logs), log_type)
        
        return jsonify([log.to_dict() for log in logs]), 200
    except SQLAlchemyError as e:
        return jsonify({"error": f"Database error: {str(e)}"}), 500


@logging_bp.route("<int:log_id>", methods=["GET"])
def get_log(log_id):
    """
    Get a specific log record by ID.
    
    Args:
        log_id: Log record ID.
    
    Returns:
        JSON Logging object or 404.
    """
    try:
        log = Logging.query.get(log_id)
        if not log:
            return jsonify({"error": "Log record not found"}), 404
        logger.info("Fetched audit log id=%s type=%s", log.id, log.log_type)
        return jsonify(log.to_dict()), 200
    except SQLAlchemyError as e:
        return jsonify({"error": f"Database error: {str(e)}"}), 500


@logging_bp.route("", methods=["POST"])
def create_log():
    """
    Create a new audit log entry.
    
    Request JSON:
        {
            "log_type": "user",  # or "system"
            "action_log": "User logged in",
            "concerned_column": "users"  # optional
        }
    
    Returns:
        JSON created Logging object or 400.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        log_type = data.get("log_type")
        action_log = data.get("action_log")
        
        # Validate required fields
        if not log_type or not action_log:
            return jsonify({"error": "log_type and action_log are required"}), 400
        
        if log_type not in ["user", "system"]:
            return jsonify({"error": "log_type must be 'user' or 'system'"}), 400
        
        log = Logging(
            log_type=log_type,
            action_log=action_log,
            concerned_column=data.get("concerned_column")
        )
        
        db.session.add(log)
        db.session.commit()
        logger.info("Created audit log id=%s type=%s", log.id, log.log_type)
        
        return jsonify(log.to_dict()), 201
    except SQLAlchemyError as e:
        db.session.rollback()
        return jsonify({"error": f"Database error: {str(e)}"}), 500


@logging_bp.route("<int:log_id>", methods=["PUT"])
def update_log(log_id):
    """
    Update a log record.
    
    Args:
        log_id: Log record ID.
    
    Request JSON:
        {
            "action_log": "User action updated"
        }
    
    Returns:
        JSON updated Logging object or 404/400.
    """
    try:
        log = Logging.query.get(log_id)
        if not log:
            return jsonify({"error": "Log record not found"}), 404
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        if "action_log" in data:
            log.action_log = data["action_log"]
        
        if "log_type" in data:
            if data["log_type"] not in ["user", "system"]:
                return jsonify({"error": "log_type must be 'user' or 'system'"}), 400
            log.log_type = data["log_type"]
        
        if "concerned_column" in data:
            log.concerned_column = data["concerned_column"]
        
        db.session.commit()
        logger.info("Updated audit log id=%s type=%s", log.id, log.log_type)
        return jsonify(log.to_dict()), 200
    except SQLAlchemyError as e:
        db.session.rollback()
        return jsonify({"error": f"Database error: {str(e)}"}), 500


@logging_bp.route("<int:log_id>", methods=["DELETE"])
def delete_log(log_id):
    """
    Delete a log record.
    
    Args:
        log_id: Log record ID.
    
    Returns:
        JSON success message or 404.
    """
    try:
        log = Logging.query.get(log_id)
        if not log:
            return jsonify({"error": "Log record not found"}), 404
        
        db.session.delete(log)
        db.session.commit()
        logger.info("Deleted audit log id=%s type=%s", log.id, log.log_type)
        return jsonify({"message": f"Log record {log_id} deleted successfully"}), 200
    except SQLAlchemyError as e:
        db.session.rollback()
        return jsonify({"error": f"Database error: {str(e)}"}), 500


@logging_bp.route("/stats", methods=["GET"])
def get_log_stats():
    """
    Get statistics about logs.
    
    Returns:
        JSON with log counts by type.
    """
    try:
        total = Logging.query.count()
        user_logs = Logging.query.filter_by(log_type="user").count()
        system_logs = Logging.query.filter_by(log_type="system").count()
        
        return jsonify({
            "total": total,
            "user_logs": user_logs,
            "system_logs": system_logs
        }), 200
    except SQLAlchemyError as e:
        return jsonify({"error": f"Database error: {str(e)}"}), 500
