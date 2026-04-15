"""Authentication and user profile controllers."""

from flask import Blueprint, request, jsonify
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from ..models import db, User


auth_bp = Blueprint("auth", __name__, url_prefix="/api/auth")
users_bp = Blueprint("users", __name__, url_prefix="/api/users")


@auth_bp.route("/signup", methods=["POST"])
def signup():
    """Create a new user account for the supervisor app."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        required = ["username", "email", "password", "first_name", "last_name"]
        for field in required:
            if not str(data.get(field, "")).strip():
                return jsonify({"error": f"{field} is required"}), 400

        username = data["username"].strip()
        email = data["email"].strip().lower()

        if User.query.filter_by(username=username).first() is not None:
            return jsonify({"error": "Username already exists"}), 409

        if User.query.filter_by(email=email).first() is not None:
            return jsonify({"error": "Email already exists"}), 409

        user = User(
            username=username,
            email=email,
            first_name=data["first_name"].strip(),
            last_name=data["last_name"].strip(),
            profile_image_path=data.get("profile_image_path"),
            is_validated=bool(data.get("is_validated", True)),
        )
        user.set_password(data["password"])

        db.session.add(user)
        db.session.commit()

        return jsonify(user.to_dict()), 201
    except IntegrityError:
        db.session.rollback()
        return jsonify({"error": "User already exists"}), 409
    except SQLAlchemyError as e:
        db.session.rollback()
        return jsonify({"error": f"Database error: {str(e)}"}), 500


@auth_bp.route("/login", methods=["POST"])
def login():
    """Authenticate user with username and password."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        username = str(data.get("username", "")).strip()
        password = str(data.get("password", "")).strip()

        if not username or not password:
            return jsonify({"error": "username and password are required"}), 400

        user = User.query.filter_by(username=username).first()
        if user is None or not user.check_password(password):
            return jsonify({"error": "Invalid credentials"}), 401

        if not user.is_validated:
            return jsonify({"error": "Account not validated"}), 403

        return jsonify({"message": "Login successful", "user": user.to_dict()}), 200
    except SQLAlchemyError as e:
        return jsonify({"error": f"Database error: {str(e)}"}), 500


@users_bp.route("/<string:username>", methods=["GET"])
def get_user(username):
    """Return public profile information for a user."""
    try:
        user = User.query.filter_by(username=username).first()
        if user is None:
            return jsonify({"error": "User not found"}), 404
        return jsonify(user.to_dict()), 200
    except SQLAlchemyError as e:
        return jsonify({"error": f"Database error: {str(e)}"}), 500
