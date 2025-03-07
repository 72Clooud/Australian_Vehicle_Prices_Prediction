from sqlalchemy import Column, Integer, String, TIMESTAMP, DECIMAL, text
from app.database.database import Base

class Prediction(Base):
    __tablename__ = 'price_predictions'
    
    prediction_id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    brand = Column(String, nullable=False)
    production_year = Column(Integer, nullable=False)
    used_or_new = Column(String, nullable=False)
    transmission = Column(String, nullable=False)
    drive_type = Column(String, nullable=False)
    fuel_type = Column(String, nullable=False)
    fuel_consumption = Column(String, nullable=False)
    kilometres = Column(Integer, nullable=False)
    cylinder_in_engine = Column(Integer, nullable=False)
    body_type = Column(String, nullable=False)
    doors = Column(Integer, nullable=False)
    seats = Column(Integer, nullable=False)
    prediction_price = Column(DECIMAL, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=text('NOW()'))
