import streamlit as st

class AppLayout:
    
    def __init__(self, title: str, discription: str):
        self.title = title
        self.discription = discription
        
    def render_header(self) -> None:
        st.title(self.title)
        st.write(self.discription)
        st.write("")
        st.write("")
    
    def render_form(self) -> dict:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            brand = st.text_input("Brand", "Toyota")
            year = st.slider("Year", 1999, 2024, 2020)
            condition = st.radio("Condition", options=["USED", "NEW", "DEMO"])
            transmission = st.radio("Transmission", options=["Automatic", "Manual"])

        with col2:
            drive_type = st.radio("Drive Type", options=["4WD", "AWD", "Front", "Other", "Rear"])
            fuel_type = st.radio("Fuel Type", options=["Diesel", "Hybrid", "LPG", "Premium", "Unleaded"])
            fuel_consumption = st.number_input("Fuel Consumption (L/100km)", min_value=1.0, step=0.1, value=6.5)
            kilometres = st.number_input("Kilometres", min_value=0, value=60000)

        with col3:
            cylinders = st.slider("Cylinders in Engine", 1, 12, 4)
            body_type = st.selectbox("Body Type", options=["SUV", "Hatchback", "Coupe", "Commercial", "Ute / Tray", "Sedan", "People Mover", "Convertible", "Wagon", "Other"])
            doors = st.slider("Number of Doors", 2, 7, 4)
            seats = st.slider("Number of Seats", 2, 12, 5)

        input_data = {
            "Brand": brand,
            "Year": year,
            "UsedOrNew": condition,
            "Transmission": transmission,
            "DriveType": drive_type,
            "FuelType": fuel_type,
            "FuelConsumption": fuel_consumption,
            "Kilometres": kilometres,
            "CylindersinEngine": cylinders,
            "BodyType": body_type,
            "Doors": doors,
            "Seats": seats
        }
        return input_data

    def render_predict(self, result: str) -> None:
        st.success(f"Prediction: {result:.2f}")