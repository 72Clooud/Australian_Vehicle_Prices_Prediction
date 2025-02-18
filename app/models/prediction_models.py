from pydantic import BaseModel, Field, model_validator
from typing import Literal

class InputData(BaseModel):
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

    @model_validator(mode="before")
    def validate_model(cls, values):
        if values["UsedOrNew"] == "NEW" and values["Kilometres"] > 0:
            raise ValueError("New cars should have 0 kilometers")
        return values
