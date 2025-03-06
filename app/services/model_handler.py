import pandas as pd
import pickle
import logging
import os

from pathlib import Path
from functools import lru_cache

class ModelHandler:
    
    def __init__(self, model_path: Path, one_hot_encoder_path: Path, label_encoder_path: Path):
        self.model_path = model_path
        self.one_hot_encoder_path = one_hot_encoder_path
        self.label_encoder_path = label_encoder_path
        self.model = None
        self.one_hot_encoder = None
        self.label_encoder = None
    
        
    # Function to load the machine learning model
    # lru_cache ensures the model is loaded only once to improve performance
    @lru_cache(maxsize=1)
    def load_model(self) -> None:
        logging.info("Loading model...")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        with open(self.model_path, 'rb') as file:
            self.model = pickle.load(file)
        logging.info("Model loaded successfully")
    
    def load_one_hot_encoder(self) -> None:
        logging.info("Loading OneHotEncoder...")
        if not os.path.exists(self.one_hot_encoder_path):
            raise FileNotFoundError(f"OneHotEncoder file not found {self.one_hot_encoder_path}")
        with open(self.one_hot_encoder_path, "rb") as file:
            self.one_hot_encoder = pickle.load(file)
        logging.info("OneHotEncoder loaded successfully")
    
    def load_label_encoder(self) -> None:
        logging.info("Loading LabelEncoder...")
        if not os.path.exists(self.label_encoder_path):
            raise FileNotFoundError(f"LabelEncoder file not found {self.label_encoder_path}")
        with open(self.label_encoder_path, "rb") as file:
            self.label_encoder = pickle.load(file)
        logging.info("LabelEncoder loaded successfully")
    
    def process_input_data(self, input_data: dict) -> pd.DataFrame:
        logging.info("Preprocessing data...")
        df = pd.DataFrame([input_data])
        
        # OneHotEncoding
        categorical_cols_for_one_hot = ["UsedOrNew", "Transmission", "DriveType", "FuelType"] 
        one_hot_encoded = self.one_hot_encoder.transform(df[categorical_cols_for_one_hot])
        one_hot_df = pd.DataFrame(one_hot_encoded, columns=self.one_hot_encoder.get_feature_names_out(categorical_cols_for_one_hot))
        df_encoded = pd.concat([df.drop(categorical_cols_for_one_hot, axis=1), one_hot_df], axis=1)
        
        # LabelEncoding
        try:
            df_encoded["Brand"] = self.label_encoder.transform(df["Brand"])
        except ValueError:
            df_encoded["Brand"] = self.label_encoder.transform([self.label_encoder.classes_[0]])
        try:
            df_encoded['BodyType'] = self.label_encoder.transform(df['BodyType'])
        except ValueError:
            df_encoded['BodyType'] = self.label_encoder.transform([self.label_encoder.classes_[0]])
        return df_encoded
    
    def predict(self, input_data):
        if self.model == None:
            self.load_model()
        if self.one_hot_encoder == None:
            self.load_one_hot_encoder()
        if self.label_encoder == None:
            self.load_label_encoder()
        
        processed_data = self.process_input_data(input_data)
        prediction = self.model.predict(processed_data)
        return prediction.tolist()
            
        