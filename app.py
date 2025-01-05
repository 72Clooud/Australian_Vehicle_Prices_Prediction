import pandas as pd
import pickle

from functools import lru_cache
from pydantic import BaseModel

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

from fastapi import FastAPI

MODEL_PATH = 'models/001_model.pkl'
ONE_HOT_ENCODER_PATH = "models/onehot_encoder.pkl"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"

app = FastAPI()

# Define the input data model using Pydantic
class InputData(BaseModel):
    Year: int
    Model: str
    UsedOrNew: str
    Transmission: str
    DriveType: str
    FuelType: str
    FuelConsumption: float
    Kilometres: int
    CylindersinEngine: int
    BodyType: str
    Doors: int
    Seats: int   

# Healthcheck endpoint to verify application status
@app.get("/healthcheck")
def healthcheck():
    return {"status": "ok"}

# Function to load the machine learning model
# lru_cache ensures the model is loaded only once to improve performance
@lru_cache(maxsize=1)
def get_model(model_path):
    print("Model Loading...")
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def preprocess_input_data(input_data):
    print("Preprocessing data...")
    
    df = pd.DataFrame([input_data])

    with open(ONE_HOT_ENCODER_PATH, "rb") as file:
        one_hot_encoder = pickle.load(file)
    
    with open(LABEL_ENCODER_PATH, "rb") as file:
        label_encoder = pickle.load(file)
    
    categorical_cols_for_one_hot = ["UsedOrNew", "Transmission", "DriveType", "FuelType"]
    one_hot_encoded = one_hot_encoder.transform(df[categorical_cols_for_one_hot])
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=one_hot_encoder.get_feature_names_out(categorical_cols_for_one_hot))
    df_encoded = pd.concat([df.drop(categorical_cols_for_one_hot, axis=1),
                            one_hot_df], axis=1)
    
    # Encode 'Model' and 'BodyType' columns using LabelEncoder
    try:
        df_encoded["Model"] = label_encoder.transform(df["Model"])
    except ValueError:
        df_encoded["Model"] = label_encoder.transform([label_encoder.classes_[0]])
    try:
        df_encoded['BodyType'] = label_encoder.transform(df['BodyType'])
    except ValueError:
        df_encoded['BodyType'] = label_encoder.transform([label_encoder.classes_[0]])
    
    return df_encoded

# Prediction endpoint to handle predictions
@app.post('/predict')
def predict(input_data: InputData):
    model = get_model(MODEL_PATH)
    df = preprocess_input_data(input_data.dict())
    pred = model.predict(df)
    return {"prediction": pred.tolist()}