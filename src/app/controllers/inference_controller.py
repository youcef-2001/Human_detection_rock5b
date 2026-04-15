"""Inference controller for human/hot object detection API."""

import logging
from io import BytesIO

import cv2
import numpy as np
from flask import Blueprint, request, jsonify

from ..services import InferenceService


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
    
    # Validate request
    if "image" not in request.files:
        return jsonify({"error": "Missing 'image' file"}), 400
    
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    
    try:
        # Read and decode image
        file_data = file.read()
        
        # Try to load as standard image first
        try:
            nparr = np.frombuffer(file_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Failed to decode image")
        except Exception:
            # Fall back to raw thermal data
            if len(file_data) % 4 != 0:
                return jsonify({"error": "Invalid thermal data format"}), 400
            
            arr = np.frombuffer(file_data, dtype="<f4")
            if arr.size != 32 * 24:
                return jsonify(
                    {"error": f"Expected 768 float32 values, got {arr.size}"}
                ), 400
            image = arr.reshape((24, 32))
        
        # Run inference
        result = _inference_service.infer(image)
        
        return jsonify({
            "human_count": result["human_count"],
            "hot_object_count": result["hot_object_count"],
            "success": True,
        }), 200
    
    except Exception as e:
        logger.error(f"Detection error: {e}", exc_info=True)
        return jsonify({"error": str(e), "success": False}), 500
