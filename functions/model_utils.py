import pandas as pd

from typing import Dict, Union

from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor


def train_model(X_train: pd.DataFrame,
                y_train: pd.Series,
                reg: XGBRegressor,
                param_grid: Dict[str, list[Union[int, float]]]) -> XGBRegressor:
    
    grid_search = GridSearchCV(estimator=reg,
                           param_grid=param_grid,
                           cv=3, n_jobs=-1, verbose=2,
                           scoring='neg_mean_squared_error')
    
    grid_search.fit(X_train, y_train)
    
    # Showing the best model
    print(f'Best params: {grid_search.best_params_}')
    print(f'Best score: {grid_search.best_score_}')
    
    best_model = grid_search.best_estimator_
    
    return best_model 


def evaluate_model(X_test: pd.DataFrame,
                   y_test: pd.Series,
                   model: XGBRegressor) -> None:
    # Predict on test set
    reg_pred = model.predict(X_test)
    
    print(f'Mean Squared Error: {mean_squared_error(y_test, reg_pred)}')
    print(f'Mean Absolute Error: {mean_absolute_error(y_test, reg_pred)}')
    print(f'Root Mean Squared Error: {root_mean_squared_error(y_test, reg_pred)}')
    print(f'R2 Score: {r2_score(y_test, reg_pred)}')