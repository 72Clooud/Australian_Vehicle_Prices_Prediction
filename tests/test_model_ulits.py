import numpy as np
import pytest

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from functions.model_utils import train_model, evaluate_model

def test_train_model():
    X_train, y_train = make_regression(n_samples=100, n_features=5, random_state=42)
    param_grid = {'n_estimators': [100, 200], 'learning_rate': [0.1, 0.2]}
    model = XGBRegressor()
    best_model = train_model(X_train, y_train, model, param_grid)
    
    assert isinstance(best_model, XGBRegressor)
    
def test_evaluate_model(capsys):
    
    X, y = np.random.rand(100, 5), np.random.rand(100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
    param_grid = {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]}
    best_model = train_model(X_train, y_train, model, param_grid)
    evaluate_model(X_test, y_test, best_model)
    captured = capsys.readouterr()
    
    assert 'Mean Squared Error:' in captured.out
    assert 'Root Mean Squared Error:' in captured.out
    assert 'Mean Absolute Error:' in captured.out
    assert 'R2 Score:' in captured.out