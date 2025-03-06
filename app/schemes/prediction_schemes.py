from pydantic import BaseModel, Field, field_validator
from typing import Literal
from datetime import datetime

class PredictionInputData(BaseModel):
    Brand: str
    Year: int = Field(..., ge=1999, le=2024)
    UsedOrNew: Literal["USED", "NEW", "DEMO"]
    Transmission: Literal["Automatic", "Manual"]
    DriveType: Literal["4WD", "AWD", "Front", "Other", "Rear"]
    FuelType: Literal["Diesel", "Hybrid", "LPG", "Premium", "Unleaded"]
    FuelConsumption: float = Field(..., ge=1)
    Kilometres: int
    CylindersinEngine: int = Field(..., ge=1, le=12)
    BodyType: str
    Doors: int = Field(..., ge=2, le=12)
    Seats: int = Field(..., ge=2, le=12)

class PredictionOutputData(BaseModel):
    prediction_price: float
    brand: str
    production_year: int = Field(..., ge=1999, le=2024)
    used_or_new: Literal["USED", "NEW", "DEMO"]
    transmission: Literal["Automatic", "Manual"]
    drive_type: Literal["4WD", "AWD", "Front", "Other", "Rear"]
    fuel_type: Literal["Diesel", "Hybrid", "LPG", "Premium", "Unleaded"]
    fuel_consumption: float = Field(..., ge=1)
    kilometres: int
    cylinder_in_engine: int = Field(..., ge=1, le=12)
    body_type: str
    doors: int = Field(..., ge=2, le=12)
    seats: int = Field(..., ge=2, le=12)
    created_at: datetime
    class Config:
        from_attributes = True