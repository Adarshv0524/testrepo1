from flask import Flask
from .app import create_app  # If using factory pattern

app = create_app()  # Initialize Flask app

# Optional: Import routes here if using separate routes.py
# from .routes import main_bp
# app.register_blueprint(main_bp)