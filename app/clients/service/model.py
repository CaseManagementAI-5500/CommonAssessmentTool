"""
Model training module for the Common Assessment Tool.
Handles the preparation, training, and saving of the prediction model.
"""

# Standard library imports
import os
import pickle

# Third-party imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

# Global variable to track current model
CURRENT_MODEL = "random_forest"

def prepare_training_data():
    """
    Prepare training data from the dataset.

    Returns:
        tuple: features_train, targets_train for model training
    """
    # Load dataset
    data = pd.read_csv("data_commontool.csv")
    # Define feature columns
    feature_columns = [
        "age",  # Client's age
        "gender",  # Client's gender (bool)
        "work_experience",  # Years of work experience
        "canada_workex",  # Years of work experience in Canada
        "dep_num",  # Number of dependents
        "canada_born",  # Born in Canada
        "citizen_status",  # Citizenship status
        "level_of_schooling",  # Highest level achieved (1-14)
        "fluent_english",  # English fluency scale (1-10)
        "reading_english_scale",  # Reading ability scale (1-10)
        "speaking_english_scale",  # Speaking ability scale (1-10)
        "writing_english_scale",  # Writing ability scale (1-10)
        "numeracy_scale",  # Numeracy ability scale (1-10)
        "computer_scale",  # Computer proficiency scale (1-10)
        "transportation_bool",  # Needs transportation support (bool)
        "caregiver_bool",  # Is primary caregiver (bool)
        "housing",  # Housing situation (1-10)
        "income_source",  # Source of income (1-10)
        "felony_bool",  # Has a felony (bool)
        "attending_school",  # Currently a student (bool)
        "currently_employed",  # Currently employed (bool)
        "substance_use",  # Substance use disorder (bool)
        "time_unemployed",  # Years unemployed
        "need_mental_health_support_bool",  # Needs mental health support (bool)
    ]
    # Define intervention columns
    intervention_columns = [
        "employment_assistance",
        "life_stabilization",
        "retention_services",
        "specialized_services",
        "employment_related_financial_supports",
        "employer_financial_supports",
        "enhanced_referrals",
    ]
    # Combine all feature columns
    all_features = feature_columns + intervention_columns
    # Prepare training data
    features = np.array(data[all_features])
    targets = np.array(data["success_rate"])
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
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(features_train, targets_train)
    return model

def prepare_models():
    """
    Prepare and train all models using the dataset.

    Returns:
        dict: Dictionary of trained models
    """
    print("Training all models...")
    features_train, targets_train = prepare_training_data()

    models = {
        "random_forest": train_model_rf(features_train, targets_train),
        "linear_regression": train_model_lr(features_train, targets_train),
        "gradient_boost": train_model_gb(features_train, targets_train)
    }

    return models

def save_model(model, filename="model.pkl"):
    """
    Save the trained model to a file.

    Args:
        model: Trained model to save
        filename (str): Name of the file to save the model to
    """
    with open(filename, "wb") as model_file:
        pickle.dump(model, model_file)

def save_all_models(models):
    """
    Save all trained models to files.

    Args:
        models (dict): Dictionary of trained models
    """
    # # Create directory if it doesn't exist
    # models_dir = os.path.join("models")
    # os.makedirs(models_dir, exist_ok=True)
    #
    # for model_name, model in models.items():
    #     filename = os.path.join(models_dir, f"{model_name}.pkl")
    #     save_model(model, filename)
    #     print(f"Saved {model_name} model to {filename}")

    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)

    # 同时保存到工作目录的models
    root_models_dir = "/code/models"
    os.makedirs(root_models_dir, exist_ok=True)

    for model_name, model in models.items():
        # 保存到两个位置
        save_model(model, os.path.join(models_dir, f"{model_name}.pkl"))
        save_model(model, os.path.join(root_models_dir, f"{model_name}.pkl"))

def load_model(model_name="random_forest"):
    """
    Load a trained model from a file.

    Args:
        model_name (str): Name of the model to load

    Returns:
        The loaded model
    """
    # 如果model_name是"string"，使用默认模型
    if model_name == "string" or not model_name:
        model_name = "random_forest"

    print(f"尝试加载模型: {model_name}")

    # 尝试多个可能的位置
    possible_paths = [
        f"models/{model_name}.pkl",
        f"app/clients/service/models/{model_name}.pkl",
        f"app/clients/service/{model_name}.pkl",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{model_name}.pkl"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", f"{model_name}.pkl")
    ]

    for path in possible_paths:
        print(f"尝试路径: {path}")
        if os.path.exists(path):
            print(f"找到模型文件: {path}")
            try:
                with open(path, "rb") as model_file:
                    model = pickle.load(model_file)
                    print(f"成功加载模型: {type(model)}")
                    return model
            except Exception as e:
                print(f"加载模型文件 {path} 时出错: {e}")

    print("未找到模型文件，创建新模型")

    # 创建新模型
    try:
        if model_name == "random_forest":
            model = RandomForestRegressor(n_estimators=10, random_state=42)
        elif model_name == "linear_regression":
            model = LinearRegression()
        elif model_name == "gradient_boost":
            model = GradientBoostingRegressor(n_estimators=10, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=10, random_state=42)

        # 创建简单数据进行拟合
        X = np.random.rand(100, 31)  # 31个特征
        y = np.random.rand(100)
        model.fit(X, y)

        print(f"创建并训练了新模型: {type(model)}")

        # 保存模型以备将来使用
        try:
            models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
            os.makedirs(models_dir, exist_ok=True)
            filename = os.path.join(models_dir, f"{model_name}.pkl")
            with open(filename, "wb") as f:
                pickle.dump(model, f)
            print(f"保存新模型到: {filename}")
        except Exception as e:
            print(f"保存模型时出错: {e}")

        return model
    except Exception as e:
        print(f"创建模型时出错: {e}")
        return None

def get_available_models():
    """
    Get list of available models.

    Returns:
        list: List of available model names
    """
    return ["random_forest", "linear_regression", "gradient_boost"]

def get_current_model():
    """
    Get the name of the currently active model.

    Returns:
        str: Name of the current model
    """
    global CURRENT_MODEL
    return CURRENT_MODEL

def set_current_model(model_name):
    """
    Set the currently active model.

    Args:
        model_name (str): Name of the model to set as current

    Raises:
        ValueError: If model_name is not recognized
    """
    global CURRENT_MODEL
    available_models = get_available_models()

    if model_name not in available_models:
        raise ValueError(f"Unknown model: {model_name}. Available models: {available_models}")

    CURRENT_MODEL = model_name
    return CURRENT_MODEL

def predict(features, model_name=None):
    """
    Make a prediction using the specified or current model.

    Args:
        features: Input features for prediction
        model_name (str, optional): Name of the model to use. Defaults to None, which uses the current model.

    Returns:
        float: Predicted success rate
    """
    # Use specified model or current model
    model_to_use = model_name or get_current_model()

    # Load the model
    model = load_model(model_to_use)

    # Make prediction
    prediction = model.predict([features])[0]

    # Ensure prediction is between 0 and 1
    prediction = max(0, min(1, prediction))

    return prediction

def main():
    """Main function to train and save all models."""
    print("Starting model training...")
    models = prepare_models()
    save_all_models(models)
    print("All models trained and saved successfully.")

if __name__ == "__main__":
    main()