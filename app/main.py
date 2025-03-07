from fastapi import FastAPI
from app.routes import prediction, user
from app.database.database import db

db.init()

app = FastAPI()

app.include_router(prediction.router)
app.include_router(user.router)
