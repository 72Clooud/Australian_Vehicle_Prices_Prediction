import pytest
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app, ModelHandler, Config

client = TestClient(app)

class TestApp:
    @pytest.fixture
    def data(self):
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

    @pytest.fixture
    def model_h(self):
        model_handler = ModelHandler(
        model_path=Config.MODEL_PATH,
        one_hot_encoder_path=Config.ONE_HOT_ENCODER_PATH,
        label_encoder_path=Config.LABEL_ENCODER_PATH,
        )
        model_handler.load_model()
        model_handler.load_one_hot_encoder()
        model_handler.load_label_encoder()
        return model_handler

    def test_health_check(self):
        response = client.get("/healthcheck")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        assert response.json() == {"status": "ok"}
            
    def test_process_input_data(self, data, model_h):
        
        model_handler = model_h        
        processed_data = model_handler.process_input_data(data)
        
        assert len(processed_data.columns) == 24, "Input data should have 24 columns"
        
        assert "DriveType_AWD" in processed_data.columns, "DriveType_AWD column missing"
        assert "UsedOrNew_USED" in processed_data.columns, "UsedOrNew_USED column missing"
        assert "Transmission_Automatic" in processed_data.columns, "Transmission_Automatic column missing"
        assert "FuelType_Diesel" in processed_data.columns, "FuelType_Diesel column missing"
        
        assert processed_data["DriveType_AWD"].iloc[0] == 1.0, "DriveType_AWD should be 1.0"
        assert processed_data["UsedOrNew_USED"].iloc[0] == 1.0, "UsedOrNew_USED should be 1.0"
        assert processed_data["Transmission_Automatic"].iloc[0] == 1.0, "Transmission_Automatic should be 1.0"
        assert processed_data["FuelType_Diesel"].iloc[0] == 1.0, "FuelType_Diesel should be 1.0"
        
    def test_predict(self, data, model_h):
        
        model_handler = model_h        
        processed_data = model_handler.process_input_data(data)
        response = client.post("/predict", json=data)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        json_response = response.json()
        assert "prediction" in json_response, "Response should contain 'prediction' field"
        assert isinstance(json_response["prediction"], list), "'prediction' should be a list"
        assert len(json_response["prediction"]) == 1, "Prediction list should have one item"
        assert isinstance(json_response["prediction"][0], (int, float)), "Prediction should be numeric"
