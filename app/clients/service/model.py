"""
Module for training and saving various machine learning models.
Includes Random Forest, Linear Regression, and Gradient Boosting models.
"""

# Standard library imports
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

# Constants
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(CURRENT_DIR, "data_commontool.csv")
MODELS_DIR = os.path.join(CURRENT_DIR, "models")

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)


def save_model(model, filename="model.pkl"):
    """
    Save the trained model to a file.

    Args:
        model: Trained model to save
        filename (str): Name of the file to save the model to
    """
    if not filename.endswith(".pkl"):
        filename = f"{filename}.pkl"

    # Save to models directory
    model_path = os.path.join(MODELS_DIR, filename)

    with open(model_path, "wb") as model_file:
        pickle.dump(model, model_file)

    print(f"Model saved to {model_path}")


def load_model(filename="model.pkl"):
    """
    Load a trained model from a file.

    Args:
        filename (str): Name of the file to load the model from

    Returns:
        The loaded model
    """
    with open(filename, "rb") as model_file:
        return pickle.load(model_file)


def prepare_training_data():
    """
    Prepare training data from CSV file.

    Returns:
        tuple: (features_train, targets_train)
    """
    # Load and preprocess the data
    data = pd.read_csv(DATA_FILE)

    # Extract features and target
    features = data.drop(columns=["success_rate"])
    targets = np.array(data["success_rate"])  # Changed from y to targets

    # Split the dataset
    features_train, _, targets_train, _ = train_test_split(
        features, targets, test_size=0.2, random_state=42
    )

    return features_train, targets_train


def train_model_rf(features_train, targets_train):
    """
    Train the Random Forest model using the dataset.

    Returns:
        RandomForestRegressor: Trained model for predicting success rates
    """
    # Initialize and train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(features_train, targets_train)

    return model


def train_model_lr(features_train, targets_train):
    """
    Train the Linear Regression model using the dataset.

    Returns:
        LinearRegression: Trained model for predicting success rates
    """
    # Initialize and train the model
    model = LinearRegression()
    model.fit(features_train, targets_train)

    return model


def train_model_gb(features_train, targets_train):
    """
    Train the Gradient Boost model using the dataset.

    Returns:
        GradientBoostingRegressor: Trained model for predicting success rates
    """
    # Initialize and train the model
    model = GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, random_state=42
    )
    model.fit(features_train, targets_train)

    return model


def train_model_mlp(features_train, targets_train):
    """
    Train the Neural Network (MLP) model using the dataset.

    Returns:
        MLPRegressor: Trained model for predicting success rates
    """
    # Initialize and train the model
    model = MLPRegressor(
        hidden_layer_sizes=(100, 50),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        batch_size="auto",
        learning_rate="constant",
        max_iter=500,
        random_state=42,
    )
    model.fit(features_train, targets_train)

    return model


def main():
    """Main function to train and save the model."""
    features_train, targets_train = prepare_training_data()

    print("Starting model training...")

    # Train the models
    model_rf = train_model_rf(features_train, targets_train)
    model_lr = train_model_lr(features_train, targets_train)
    model_gb = train_model_gb(features_train, targets_train)

    # Save the models
    save_model(model_rf, filename="random_forest.pkl")
    save_model(model_lr, filename="linear_regression.pkl")
    save_model(model_gb, filename="gradient_boost.pkl")

    print("Model training completed and saved successfully.")


if __name__ == "__main__":
    main()
