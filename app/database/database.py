from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core.config import settings
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class DatabaseSession:
    def __init__(self):
        self._engine = create_engine(
            f"postgresql://{settings.db_user}:{settings.db_password}@{settings.db_hostname}:{settings.db_port}/{settings.db_name}",
            echo=True
        )

        self._SessionLocal = sessionmaker(
            bind=self._engine,
            autoflush=False,
            autocommit=False,
            expire_on_commit=False
        )
       
    def init(self):
        Base.metadata.create_all(bind=self._engine)
    
    def get_session(self):
        return self._SessionLocal()
        
db = DatabaseSession()