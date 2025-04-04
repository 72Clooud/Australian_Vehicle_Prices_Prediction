from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import prediction, user, auth
from app.database.database import db

db.init()

app = FastAPI()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(prediction.router)
app.include_router(user.router)
app.include_router(auth.router)

@app.get("/")
async def root():
    return {"message": "root"}