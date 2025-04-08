"""
Model factory module for managing multiple machine learning models.
Provides functionality to load and switch between different models.
"""

import os
import pickle
from typing import List, Optional
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

# Constants
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
AVAILABLE_MODELS = ["random_forest", "linear_regression", "gradient_boost"]
DEFAULT_MODEL = "random_forest"

# Global variable to store current model name
current_model_name = DEFAULT_MODEL

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)


def get_model_path(model_name: str) -> str:
    """
    Get the file path for a model.

    Args:
        model_name: Name of the model

    Returns:
        str: Path to the model file
    """
    return os.path.join(MODELS_DIR, f"{model_name}.pkl")


def load_model(model_name: Optional[str] = None):
    """
    Load a model from file. If model doesn't exist, create and save it first using real data.

    Args:
        model_name: Name of the model to load (default: current model)

    Returns:
        The loaded model
    """
    if model_name is None:
        model_name = current_model_name

    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Model '{model_name}' not found. Available models: {AVAILABLE_MODELS}"
        )

    model_path = get_model_path(model_name)

    # If model doesn't exist, automatically create and save it
    if not os.path.exists(model_path):
        print(
            f"Model file not found: {model_path}. Creating a new model using real data..."
        )
        # Ensure directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Load real data from CSV
        import pandas as pd

        # Since both files are in the same directory, we can use a simple relative path
        data_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data_commontool.csv"
        )

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        data = pd.read_csv(data_path)

        # Extract features and target
        X = data.drop(columns=["success_rate"])
        y = data["success_rate"]

        print(
            f"Using real data for training: {X.shape[0]} samples, {X.shape[1]} features"
        )

        # Choose the appropriate model type based on model name
        if model_name == "random_forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_name == "linear_regression":
            model = LinearRegression()
        elif model_name == "gradient_boost":
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)

        # Train and save the model
        model.fit(X, y)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"Created and saved new model to {model_path}")

        return model

    # Normal model loading
    with open(model_path, "rb") as model_file:
        return pickle.load(model_file)


def set_current_model(model_name: str) -> str:
    """
    Set the current model to use for predictions.

    Args:
        model_name: Name of the model to set as current

    Returns:
        str: Name of the newly set current model
    """
    global current_model_name

    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Model '{model_name}' not found. Available models: {AVAILABLE_MODELS}"
        )

    current_model_name = model_name
    return current_model_name


def get_current_model_name() -> str:
    """
    Get the name of the currently selected model.

    Returns:
        str: Name of the current model
    """
    return current_model_name


def get_available_models() -> List[str]:
    """
    Get a list of all available models.

    Returns:
        List[str]: List of available model names
    """
    return AVAILABLE_MODELS
