from app.ml_models.logistic_model import LogisticRegressionModel
from app.ml_models.random_forest_model import RandomForestModel
from app.ml_models.decision_tree_model import DecisionTreeModel

class ModelManager:
    def __init__(self):
        self.models = {
            "logistic": LogisticRegressionModel(),
            "random_forest": RandomForestModel(),
            "decision_tree": DecisionTreeModel()
        }
        self.current_model_name = "random_forest"

    def get_available_models(self):
        return list(self.models.keys())

    def get_current_model(self):
        return self.current_model_name

    def switch_model(self, name: str):
        if name not in self.models:
            return False
        self.current_model_name = name
        return True

    def predict(self, input_data):
        return self.models[self.current_model_name].predict(input_data)

model_manager = ModelManager()
