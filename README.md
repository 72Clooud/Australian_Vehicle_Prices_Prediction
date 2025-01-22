# Australian Vehicle Prices Prediction 🚗💰
## Project Overview
The Australian Vehicle Prices Prediction project is designed to predict the prices of vehicles in Australia based on a variety 
of features such as brand, year of production, fuel type, and more. This regression model aims to provide accurate price 
estimates, making it valuable for car dealerships, online marketplaces, and individual users seeking to evaluate vehicle prices.
## Features and Dataset 📊
The model is trained on the following features:
- Brand: Vehicle manufacturer (e.g., Toyota, Ford).
- Year: Year of production (integer between 1999 and 2024).
- UsedOrNew: Condition of the vehicle ("USED", "NEW", or "DEMO").
- Transmission: Type of transmission ("Automatic" or "Manual").
- DriveType: Drive configuration ("4WD", "AWD", "Front", "Other", "Rear").
- FuelType: Type of fuel used ("Diesel", "Hybrid", "LPG", "Premium", "Unleaded").
- FuelConsumption: Fuel consumption (float, minimum 1 L/100km).
- Kilometres: Total distance the vehicle has traveled (integer).
- CylindersinEngine: Number of cylinders in the engine (integer between 1 and 12).
- BodyType: Vehicle body type (e.g., Sedan, SUV).
- Doors: Number of doors (integer between 2 and 12).
- Seats: Number of seats (integer between 2 and 12).
### Dataset 📂
- Source: https://www.kaggle.com/datasets/nelgiriyewithana/australian-vehicle-prices/data
- Format: The original dataset is in CSV format. For prediction, input data is provided in JSON format.
## Technology Stack 🛠️
- Programming Language: Python 🐍
- Data Processing: Pandas, NumPy
- Model Training: XGBoost, scikit-learn
- Hyperparameter Tuning: GridSearchCV
- Metrics: mean_squared_error (MSE), mean_absolute_error (MAE), root_mean_squared_error (RMSE), R²
- Model Serving: FastAPI 🚀
- Testing Framework: pytest 🧪
## Machine Learning Model 🤖
- Algorithm: XGBoost Regressor
- Hyperparameter Tuning: Grid Search was used to optimize the model parameters.
- Evaluation Metrics: MSE MAE RMSE R²
## Installation and Usage 📥
### Prerequisites ✅
- Python 3.8 or higher
- Recommended system configuration: 8GB RAM or more
### Installation 🛠️
1. Clone the repository:
```bash
git clone <repository-link>
cd australian-vehicle-prices-prediction
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
### Running the Application 🚀
1. Start the FastAPI server:
```bash
uvicorn main:app --reload
```
2. Open your browser and navigate to:
```bash
http://127.0.0.1:8000/docs
```
This will open the interactive Swagger UI for testing the API.
3. Use the /predict endpoint to submit vehicle data in JSON format and receive the predicted price.
### Example JSON Input ✍️
```json
{
  "Brand": "Toyota",
  "Year": 2020,
  "UsedOrNew": "USED",
  "Transmission": "Automatic",
  "DriveType": "AWD",
  "FuelType": "Unleaded",
  "FuelConsumption": 8.5,
  "Kilometres": 45000,
  "CylindersinEngine": 4,
  "BodyType": "SUV",
  "Doors": 4,
  "Seats": 5
}
```
### Running with Docker 🐳
1. Build the Docker image:
```docker
docker build -t australian-vehicle-prices .
```
2. Run the Docker container:
```docker
docker run -p 8000:8000 australian-vehicle-prices
```
3. Open your browser and navigate to:
```docker
http://127.0.0.1:8000/docs
```
You can test the API directly from the Swagger UI.
## Project Structure 🗂️
- main.py: Contains the FastAPI application and API endpoints.
- model/: Directory for the trained XGBoost model.
- data/: Contains sample data.
- scripts/: Python scripts for training and evaluating machine learning model.
- notebooks/: Jupyter notebooks for exploratory data analysis (EDA) and model training.
- tests/: Unit tests for the application and utilities.
- requirements.txt: List of dependencies required to run the project.
## License 📄
This project is licensed under the MIT License. See the LICENSE file for details.
