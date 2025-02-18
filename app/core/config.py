from pathlib import Path

class Config:
    BASE_DIR = Path(__file__).resolve().parent.parent
    MODEL_PATH = BASE_DIR.parent / "ml_models" / "xgb_model.pkl"
    ONE_HOT_ENCODER_PATH = BASE_DIR.parent / "ml_models" / "OneHot_encoder.pkl"
    LABEL_ENCODER_PATH = BASE_DIR.parent / "ml_models" / "Label_encoder.pkl"
