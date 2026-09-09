"""
Vectorless RAG — Flask Application Factory
============================================
Creates and configures the Flask application instance.
Registers blueprints, error handlers, and initializes services.
"""

import os
from flask import Flask, jsonify
from config import config_map
from app.utils.logger import setup_logging
from app.utils.exceptions import VectorlessRAGError


def create_app(config_name="default"):
    """
    Application factory function.

    Args:
        config_name: Name of the configuration to use (development/production/testing).

    Returns:
        Configured Flask application instance.
    """
    app = Flask(__name__)

    # --- Load Configuration ---
    config_class = config_map.get(config_name, config_map["default"])
    app.config.from_object(config_class)

    # Run config-specific initialization (e.g., create directories)
    config_class.init_app(app)

    # --- Setup Logging ---
    setup_logging(app)
    app.logger.info(f"Application starting with '{config_name}' configuration.")

    # --- Register Blueprints ---
    _register_blueprints(app)

    # --- Register Error Handlers ---
    _register_error_handlers(app)

    # --- Log startup diagnostics ---
    _log_startup_info(app)

    return app


def _register_blueprints(app):
    """Register all Flask blueprints (route modules)."""
    from app.routes.main import main_bp
    from app.routes.api import api_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp, url_prefix="/api")

    app.logger.info("Blueprints registered: main, api")


def _register_error_handlers(app):
    """Register global error handlers for consistent JSON API responses."""

    @app.errorhandler(VectorlessRAGError)
    def handle_app_error(error):
        """Handle custom application errors."""
        response = {
            "success": False,
            "error": {
                "type": error.__class__.__name__,
                "message": str(error),
            }
        }
        return jsonify(response), error.status_code if hasattr(error, "status_code") else 500

    @app.errorhandler(404)
    def handle_not_found(error):
        """Handle 404 errors."""
        return jsonify({
            "success": False,
            "error": {"type": "NotFound", "message": "The requested resource was not found."}
        }), 404

    @app.errorhandler(413)
    def handle_file_too_large(error):
        """Handle file size exceeded errors."""
        max_mb = app.config.get("MAX_CONTENT_LENGTH", 0) // (1024 * 1024)
        return jsonify({
            "success": False,
            "error": {"type": "FileTooLarge", "message": f"File exceeds maximum size of {max_mb}MB."}
        }), 413

    @app.errorhandler(500)
    def handle_internal_error(error):
        """Handle unexpected internal errors."""
        app.logger.error(f"Internal server error: {error}", exc_info=True)
        return jsonify({
            "success": False,
            "error": {"type": "InternalError", "message": "An unexpected error occurred."}
        }), 500


def _log_startup_info(app):
    """Log useful startup diagnostics."""
    nvidia_key = app.config.get("NVIDIA_API_KEY", "")
    key_status = "SET" if nvidia_key and nvidia_key != "your_nvidia_api_key_here" else "NOT SET"

    app.logger.info(f"NVIDIA API Key: {key_status}")
    app.logger.info(f"LLM Model: {app.config.get('LLM_MODEL')}")
    app.logger.info(f"Upload folder: {app.config.get('UPLOAD_FOLDER')}")
    app.logger.info(f"Index folder: {app.config.get('INDEX_FOLDER')}")
    app.logger.info(f"Max traversal depth: {app.config.get('MAX_TRAVERSAL_DEPTH')}")
