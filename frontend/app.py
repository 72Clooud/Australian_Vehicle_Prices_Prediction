import streamlit as st
from ui.layout import AppLayout
from services.api_handler import ApiHandler

base_url = "http://127.0.0.1:8000"

class StreamlitApp:
    
    def __init__(self):
        self.layout = AppLayout(title ="Vehicle Prices Prediction 🚗💰",
                   discription="Application for predicting vehicle prices based on input features. Select the appropriate information about your car and press predict 😊")
        self.api_handler = ApiHandler(base_url)

    def get_form_data(self) -> dict:
        input_data = self.layout.render_form()
        return input_data
    
    def load_web_page(self) -> None:
        self.layout.render_header()

    def get_predict(self, input_data: dict):
        result = self.api_handler.get_prediction(input_data)
        return result
    
    def load_prediction(self, result) -> None:
        self.layout.render_predict(result)
        
if __name__ == "__main__":
    
    streamlit_app = StreamlitApp()
    streamlit_app.load_web_page()
    form_data = streamlit_app.get_form_data()
    
    if st.button("Predict", icon="🔥"):
        st.write("")
        response = streamlit_app.get_predict(form_data)
        if "error" in response:
            st.error("Comunication error with API")
        else:
            prediction_list = response.get("prediction", [])
            if prediction_list:
                prediction = prediction_list[0]
                streamlit_app.load_prediction(prediction)
            else:
                st.error("No prediction returned by the model")