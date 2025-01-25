import requests

class ApiHandler:
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        
    def get_prediction(self, input_data: dict) -> dict:
        endpoint = f"{self.base_url}/predict"
        try:
            response = requests.post(endpoint, json=input_data)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}
        