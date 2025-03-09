from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from typing import ClassVar

load_dotenv()

class Settings(BaseSettings):
    BASE_DIR: ClassVar = Path(__file__).resolve().parent.parent
    MODEL_PATH: Path = BASE_DIR.parent / "ml_models" / "xgb_model.pkl"
    ONE_HOT_ENCODER_PATH: Path = BASE_DIR.parent / "ml_models" / "OneHot_encoder.pkl"
    LABEL_ENCODER_PATH: Path = BASE_DIR.parent / "ml_models" / "Label_encoder.pkl"
    
    db_user: str
    db_password: str
    db_hostname: str
    db_port: str
    db_name: str
    
    secret_key: str
    algorithm: str
    access_token_expire_minutes: int
    
    class Config:
        env_file = ".env"
        
settings = Settings()