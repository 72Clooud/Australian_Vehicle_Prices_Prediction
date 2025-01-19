import pytest
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from scripts.train import ModelHandler

class TestModelHandler:

    @pytest.fixture()
    def sample_data(self):
        X = pd.DataFrame({
        "feature1": np.random.rand(100),
        "feature2": np.random.rand(100)
        })
        y = pd.Series(np.random.rand(100))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    @pytest.fixture()
    def setup(self):
        
        model = XGBRegressor(n_estimators=1000, objective='reg:squarederror', random_state=1234)
        param_grid = {
            'n_estimators': [500, 1000],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'gamma': [0, 0.1, 0.2],
        }
        
        model_handler = ModelHandler(model)
        
        return model_handler, param_grid
    
    def test_train(self, sample_data, setup):
        
        X_train, X_test, y_train, y_test = sample_data
        model_handler, param_grid = setup
        
        model = model_handler.train(X_train, y_train, param_grid)
        
        assert hasattr(model, 'n_estimators')
        assert hasattr(model, 'max_depth')
        
    def test_evaluate(self, sample_data, setup, capsys):
        X_train, X_test, y_train, y_test = sample_data
        model_handler, param_grid = setup
        
        model = model_handler.train(X_train, y_train, param_grid)
        model_handler.evaluate(X_test, y_test)

        captured = capsys.readouterr()

        predictions = model_handler.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        rmse = root_mean_squared_error(y_test, predictions) 
        r2 = r2_score(y_test, predictions)

        assert f"Mean Squared Error: {mse}" in captured.out
        assert f"Mean Absolute Error: {mae}" in captured.out
        assert f"Root Mean Squared Error: {rmse}" in captured.out
        assert f"R2 Score: {r2}" in captured.out