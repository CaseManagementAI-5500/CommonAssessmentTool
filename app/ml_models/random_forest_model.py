from app.ml_models.base_model import BaseModel
from app.clients.service.logic import interpret_and_calculate

# Just a placeholder for now
class RandomForestModel(BaseModel):
    def predict(self, input_data: dict):
        return interpret_and_calculate(input_data)
