from sqlalchemy import Column, Integer, String, DateTime, DECIMAL
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime, timezone

Base = declarative_base()

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
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
        
    def __repr__(self):
        return (f"Prediction(ID: {self.prediction_id}, "
                f"Brand: {self.brand}, "
                f"Year: {self.production_year}, "
                f"Condition: {self.used_or_new}, "
                f"Transmission: {self.transmission}, "
                f"Drive: {self.drive_type}, "
                f"Fuel: {self.fuel_type}, "
                f"Consumption: {self.fuel_consumption} L/100km, "
                f"Cylinders: {self.cylinder_in_engine}, "
                f"Body: {self.body_type}, "
                f"Doors: {self.doors}, "
                f"Seats: {self.seats}, "
                f"Predicted Price: {self.prediction_price}, "
                f"Timestamp: {self.timestamp})")
