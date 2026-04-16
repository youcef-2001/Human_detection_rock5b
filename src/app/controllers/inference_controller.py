"""Inference controller for human/hot object detection API."""

import json
import logging

import cv2
import numpy as np
from flask import Blueprint, request, jsonify

from ..services import InferenceService
from ..services.inference_service import decode_npy_payload


logger = logging.getLogger(__name__)

inference_bp = Blueprint("inference", __name__, url_prefix="/inference")
_inference_service: "InferenceService" = None


def init_inference_service(service: InferenceService) -> None:
    """
    Initialize inference service for controller.
    
    Args:
        service: Configured InferenceService instance.
    """
    global _inference_service
    _inference_service = service


@inference_bp.route("/detect", methods=["POST"])
def detect():
    """
    Detect humans and hot objects in provided image.
    
    Expected request:
        - Content-Type: multipart/form-data
        - File field: 'image' (JPEG, PNG, or binary thermal data)
    
    Returns:
        JSON with detection counts:
        {
            "human_count": int,
            "hot_object_count": int,
            "success": bool
        }
    
    Status codes:
        200: Detection successful
        400: Invalid request or file
        500: Detection failed
    """
    if _inference_service is None:
        logger.error("Inference service not initialized")
        return jsonify({"error": "Service not available"}), 500

    if hasattr(_inference_service, "is_available") and not _inference_service.is_available():
        detail = (
            _inference_service.get_init_error()
            if hasattr(_inference_service, "get_init_error")
            else "Inference backend unavailable"
        )
        return jsonify({"error": detail, "success": False}), 503
    
    try:
        image = None

        if "image" in request.files:
            file = request.files["image"]
            if file.filename == "":
                return jsonify({"error": "No file selected"}), 400

            file_data = file.read()
            if not file_data:
                return jsonify({"error": "Empty file payload"}), 400

            # Try as regular image first.
            nparr = np.frombuffer(file_data, np.uint8)
            decoded = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if decoded is not None:
                image = decoded
            else:
                # Fall back to thermal payload (raw float32 or npy bytes).
                try:
                    image = decode_npy_payload(file_data)
                except ValueError as decode_error:
                    return jsonify({"error": str(decode_error)}), 400
                if image.size != 32 * 24:
                    return jsonify(
                        {"error": f"Expected 768 float32 values, got {image.size}"}
                    ), 400

        elif request.is_json:
            # Support websocket-like JSON payload for easier API integrations.
            payload = request.get_json(silent=True)
            if payload is None:
                return jsonify({"error": "Invalid JSON payload"}), 400
            try:
                image = decode_npy_payload(json.dumps(payload))
            except ValueError as decode_error:
                return jsonify({"error": str(decode_error)}), 400
            if image.size != 32 * 24:
                return jsonify(
                    {"error": f"Expected 768 float32 values, got {image.size}"}
                ), 400
        else:
            return jsonify({"error": "Missing 'image' file or JSON payload"}), 400
        
        # Run inference
        result = _inference_service.infer(image)
        logger.info(f"Inference result: {result}")
        return jsonify({
            "human_count": result["human_count"],
            "hot_object_count": result["hot_object_count"],
            "success": True,
        }), 200
    
    except Exception as e:
        logger.error(f"Detection error: {e}", exc_info=True)
        return jsonify({"error": str(e), "success": False}), 500
