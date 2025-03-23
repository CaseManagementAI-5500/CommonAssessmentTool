# app/ml/model_manager.py
import joblib

class ModelManager:
    _available_models = {
        "logistic_regression": "app/ml/models/logistic_model.pkl",
        "random_forest": "app/ml/models/rf_model.pkl",
        "xgboost": "app/ml/models/xgb_model.pkl"
    }

    _loaded_models = {}
    _current_model_name = "logistic_regression"

    @classmethod
    def list_models(cls):
        return list(cls._available_models.keys())

    @classmethod
    def get_current_model(cls):
        return cls._current_model_name

    @classmethod
    def switch_model(cls, model_name: str):
        if model_name not in cls._available_models:
            raise ValueError(f"Model '{model_name}' not available.")
        cls._current_model_name = model_name

    @classmethod
    def predict(cls, input_data):
        model_name = cls._current_model_name
        if model_name not in cls._loaded_models:
            model_path = cls._available_models[model_name]
            cls._loaded_models[model_name] = joblib.load(model_path)
        model = cls._loaded_models[model_name]
        return model.predict(input_data)
