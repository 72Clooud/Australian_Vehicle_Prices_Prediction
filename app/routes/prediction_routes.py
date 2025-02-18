from fastapi import APIRouter
from models.prediction_models import InputData
from services.model_handler import ModelHandler
from core.config import Config

# FastAPI application setup
router = APIRouter()

# Initialize the model handler
model_handler = ModelHandler(
    model_path=Config.MODEL_PATH,
    one_hot_encoder_path=Config.ONE_HOT_ENCODER_PATH,
    label_encoder_path=Config.LABEL_ENCODER_PATH,
)

# Healthcheck endpoint to verify application status
@router.get("/healthcheck")
def healthcheck():
    return {"status": "ok"}

# Prediction endpoint
@router.post('/predict')
def predict(input_data: InputData):
    prediction = model_handler.predict(input_data.model_dump())
    return {"prediction": prediction}