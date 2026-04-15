"""Hello World controller for basic API testing."""

from flask import Blueprint

hello_bp = Blueprint("hello", __name__, url_prefix="/hello")


@hello_bp.route("/", methods=["GET"])
def hello_world():
    """
    Simple Hello World endpoint for API health check.
    
    Returns:
        JSON response with greeting message.
    """
    return {"message": "Hello World"}, 200
