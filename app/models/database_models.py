from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class Prediction(Base):
    __tablename__ = 'price_predictions'
    
    prediction_id = Column(Integer, primary_key=True, index=True)
    brand = Column(String)
    production_year = Column(Integer)
    used_or_new = Column(String)
    transmission = Column(String)
    drive_type = Column(String)
    fuel_type = Column(String)
    fuel_consumption = Column(String)
    kilometres = Column(Integer)
    cylinder_in_engine = Column(Integer)
    body_type = Column(String)
    doors = Column(Integer)
    seats = Column(Integer)
    prediction_price = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)
        
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
