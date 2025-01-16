import pytest
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app, preprocess_input_data

client = TestClient(app)

@pytest.fixture
def data():
    return {
        "Brand": "Toyota",
        "Year": 2015,
        "UsedOrNew": "USED",
        "Transmission": "Automatic",
        "DriveType": "AWD",
        "FuelType": "Diesel",
        "FuelConsumption": 7.5,
        "Kilometres": 80000,
        "CylindersinEngine": 4,
        "BodyType": "Sedan",
        "Doors": 4,
        "Seats": 5,
        "Price": "20000"
        }

def test_health_check():
    response = client.get("/healthcheck")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    assert response.json() == {"status": "ok"}
        
def test_processing_input_data(data):
    
    input_data = preprocess_input_data(data)
    
    assert len(input_data.columns) == 24, "Input data should have 24 columns"
    
    assert "DriveType_AWD" in input_data.columns, "DriveType_AWD column missing"
    assert "UsedOrNew_USED" in input_data.columns, "UsedOrNew_USED column missing"
    assert "Transmission_Automatic" in input_data.columns, "Transmission_Automatic column missing"
    assert "FuelType_Diesel" in input_data.columns, "FuelType_Diesel column missing"
    
    assert input_data["DriveType_AWD"].iloc[0] == 1.0, "DriveType_AWD should be 1.0"
    assert input_data["UsedOrNew_USED"].iloc[0] == 1.0, "UsedOrNew_USED should be 1.0"
    assert input_data["Transmission_Automatic"].iloc[0] == 1.0, "Transmission_Automatic should be 1.0"
    assert input_data["FuelType_Diesel"].iloc[0] == 1.0, "FuelType_Diesel should be 1.0"
    
def test_predict(data):
    response = client.post("/predict", json=data)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    json_response = response.json()
    assert "prediction" in json_response, "Response should contain 'prediction' field"
    assert isinstance(json_response["prediction"], list), "'prediction' should be a list"
    assert len(json_response["prediction"]) == 1, "Prediction list should have one item"
    assert isinstance(json_response["prediction"][0], (int, float)), "Prediction should be numeric"
