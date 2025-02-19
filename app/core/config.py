from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    BASE_DIR = Path(__file__).resolve().parent.parent
    MODEL_PATH = BASE_DIR.parent / "ml_models" / "xgb_model.pkl"
    ONE_HOT_ENCODER_PATH = BASE_DIR.parent / "ml_models" / "OneHot_encoder.pkl"
    LABEL_ENCODER_PATH = BASE_DIR.parent / "ml_models" / "Label_encoder.pkl"
    
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv('DB_PASSWORD')
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT")
    DB_NAME = os.getenv("DB_NAME")