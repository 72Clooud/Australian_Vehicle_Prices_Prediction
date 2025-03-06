from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core.config import Config
from app.models.prediction_models import Base

class DatabaseSession:
    def __init__(self):
        self._engine = create_engine(
            f"postgresql://{Config.DB_USER}:{Config.DB_PASSWORD}@{Config.DB_HOST}:{Config.DB_PORT}/{Config.DB_NAME}",
            echo=True
        )

        self._SessionLocal = sessionmaker(
            bind=self._engine,
            autoflush=False,
            expire_on_commit=False
        )
       
    def init(self):
        Base.metadata.create_all(bind=self._engine)
    
    def get_session(self):
        return self._SessionLocal()
        
        
db = DatabaseSession()