from fastapi import FastAPI
from app.routes.prediction_routes import router as prediction_router
from app.database.database import db

db.init()

app = FastAPI()

app.include_router(prediction_router)
