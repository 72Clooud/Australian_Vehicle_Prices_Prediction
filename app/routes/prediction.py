from fastapi import APIRouter, Depends, status, HTTPException, Response
from app.schemes.prediction import PredictionInputData, PredictionOutputData
from app.services.model_handler import ModelHandler
from app.core.config import settings
from app.database.dependencis import get_db
from app.models.prediction import Prediction
from app.auth.oauth2 import get_current_user
from sqlalchemy.orm import Session
from typing import List

# FastAPI application setup
router = APIRouter(
    prefix="/predict",
    tags=["Prediction"]
)

@router.on_event("startup")
def init_model():
    global model_handler
    model_handler = ModelHandler(
        model_path=settings.MODEL_PATH,
        one_hot_encoder_path=settings.ONE_HOT_ENCODER_PATH,
        label_encoder_path=settings.LABEL_ENCODER_PATH,
    )

# Healthcheck endpoint to verify application status
@router.get("/healthcheck")
def healthcheck():
    return {"status": "ok"}

# Prediction endpoint
@router.post('/',status_code=status.HTTP_201_CREATED)
def predict(input_data: PredictionInputData, db: Session = Depends(get_db), current_user: int = Depends(get_current_user)):
    prediction = model_handler.predict(input_data.model_dump())
    db_prediction = Prediction(
        brand=input_data.Brand,
        production_year=input_data.Year,
        used_or_new=input_data.UsedOrNew,
        transmission=input_data.Transmission,
        drive_type=input_data.DriveType,
        fuel_type=input_data.FuelType,
        fuel_consumption=input_data.FuelConsumption,
        kilometres=input_data.Kilometres,
        cylinder_in_engine=input_data.CylindersinEngine,
        body_type=input_data.BodyType,
        doors=input_data.Doors,
        seats=input_data.Seats,
        prediction_price=round(prediction[0], 2),
        owner_id=current_user.id
    )
    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)
    return {"prediction": prediction}

@router.get('/', response_model=List[PredictionOutputData])
def get_predictions(db: Session = Depends(get_db), current_user: int = Depends(get_current_user)):
    predictions = db.query(Prediction).filter(Prediction.owner_id == current_user.id).all()
    if not predictions:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"There is no prediction for {current_user.email} user")
    return predictions

@router.get("/{id}", response_model=PredictionOutputData)
def get_one_prediction(id: int, db: Session = Depends(get_db), current_user: int = Depends(get_current_user)):
    prediction = db.query(Prediction).filter(Prediction.prediction_id == id, Prediction.owner_id == current_user.id).first()
    if not prediction:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Prediction with {id} does not exist")
    return prediction

@router.delete("/{id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_prediction(id: int, db: Session = Depends(get_db), current_user: int = Depends(get_current_user)):
    prediction_query = db.query(Prediction).filter(Prediction.prediction_id == id)
    prediction = prediction_query.first()
    if prediction is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="Prediction with {id} does not exist")
    prediction_query.delete(synchronize_session=False)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
    