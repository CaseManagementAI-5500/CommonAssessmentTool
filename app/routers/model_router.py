from fastapi import APIRouter, HTTPException
from app.ml.model_manager import ModelManager

router = APIRouter(prefix="/model", tags=["Model Management"])

@router.get("/list")
def list_models():
    return {"available_models": ModelManager.list_models()}

@router.get("/current")
def current_model():
    return {"current_model": ModelManager.get_current_model()}

@router.post("/switch")
def switch_model(model_name: str):
    try:
        ModelManager.switch_model(model_name)
        return {"message": f"Switched to model: {model_name}"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
