"""
Script to initialize and save all available models.
Run this script to train and save models before starting the application.
"""

from app.clients.service.model import main as train_models

if __name__ == "__main__":
    print("Initializing models...")
    train_models()
    print("Models initialized successfully!")