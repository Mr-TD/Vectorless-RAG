"""
Vectorless RAG — Application Entry Point
==========================================
Run this file to start the Flask development server.

Usage:
    python run.py
"""

import os
from app import create_app

# Determine environment from env var, default to 'development'
config_name = os.getenv("FLASK_ENV", "development")
app = create_app(config_name)

if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_PORT", 5000))
    debug = app.config.get("DEBUG", True)

    print(f"\n{'='*60}")
    print(f"  Vectorless RAG Server")
    print(f"  Environment : {config_name}")
    print(f"  URL         : http://localhost:{port}")
    print(f"  Debug       : {debug}")
    print(f"{'='*60}\n")

    app.run(host=host, port=port, debug=debug)
