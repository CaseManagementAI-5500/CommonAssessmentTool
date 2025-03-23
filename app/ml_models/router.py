from fastapi import APIRouter, HTTPException, Depends
from app.ml_models.model_manager import model_manager
from app.clients.schema import PredictionInput
from app.auth.router import get_current_user

router = APIRouter(prefix="/ml", tags=["ml-models"])

@router.get("/models")
def list_models():
    return {"available_models": model_manager.get_available_models()}

@router.get("/model")
def current_model():
    return {"current_model": model_manager.get_current_model()}

@router.post("/model")
def switch_model(model_name: str):
    if not model_manager.switch_model(model_name):
        raise HTTPException(status_code=404, detail="Model not found")
    return {"message": f"Switched to model: {model_name}"}

@router.post("/predict")
def predict(input_data: PredictionInput, user=Depends(get_current_user)):
    return model_manager.predict(input_data.dict())
