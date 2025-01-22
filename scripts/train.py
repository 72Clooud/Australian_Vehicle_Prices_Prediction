import pandas as pd
import os
import pickle
from tqdm import tqdm
from typing import List, Union, Dict

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
from xgboost import XGBRegressor


class DataProcessor:
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def drop_unnecessary_and_NA_values(self, cols: List[str]) -> "DataProcessor":
        self.df.drop(columns=cols, inplace=True)
        self.df = self.df.apply(lambda x: x.fillna(x.value_counts().index[0]))
        return self
    
    def remove_dash_symbol(self) -> "DataProcessor":
        mask = self.df.apply(lambda col: col.astype(str).str.contains('-')).any(axis=1)
        self.df = self.df[~mask]
        return self
    
    def remove_POA_values(self, col: str = "Price") -> "DataProcessor":
        self.df = self.df[self.df[col] != 'POA'].reset_index(drop=True)            
        return self
    
    def from_cat_to_int(self, cols_for_convert: List[str]) -> "DataProcessor":
        for col in tqdm(cols_for_convert, desc="Converting categorical to int"):
            self.df[col] = self.df[col].astype(str)
            self.df[col] = self.df[col].str.replace('[^0-9]', '', regex=True)
            self.df[col] = self.df[col].astype(int)
        return self
    
    def from_cat_to_float(self, cols_convert_to_float: List[str]) -> "DataProcessor":
        for col in tqdm(cols_convert_to_float, desc="Converting categorical to float"):
            self.df[col] = self.df[col].astype(str)
            self.df[col] = self.df[col].str.extract(r'(\d+\.?\d*)').astype(float)
        return self
        
    def convert_to_int(self, cols_to_int: List[str]) -> "DataProcessor":
        for col in tqdm(cols_to_int, desc="Converting to int"):
            self.df[col] = self.df[col].astype(int)
        return self
    
    def remove_outliers(self, outliers_vals: dict) -> "DataProcessor":
        for col, (min_val, max_val) in outliers_vals.items():
            self.df = self.df[(self.df[col] >= min_val) & (self.df[col] <= max_val)]
        self.df.reset_index(drop=True, inplace=True)
        return self
    
    def get_dataframe(self) -> pd.DataFrame:
        return self.df
    
class SaveModel:
    
    def __init__(self, path: str):
        self.path = path
        
    def save_with_pickle(self, model, file_path) -> pickle:
        folder = os.path.dirname("../models")
        if not os.path.exists(folder):
            os.mkdir("../models")
        with open(file_path, "wb") as file:
            pickle.dump(model, file)
        return self
    
class Encoder(SaveModel):
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def one_hot_encode(self, cols: List[str], save_model: bool = True) -> "Encoder":
        one_hot_encoder = OneHotEncoder(sparse_output=False)
        one_hot_encoded = one_hot_encoder.fit_transform(self.df[cols])
        one_hot_df = pd.DataFrame(
            one_hot_encoded, columns=one_hot_encoder.get_feature_names_out(cols), index=self.df.index
        ) 
        self.df = pd.concat([self.df.drop(cols, axis=1),
                                one_hot_df], axis=1)
        if save_model:
            self.save_with_pickle(one_hot_encoder, "./models/3OneHot_encoder.pkl")
        return self
    
    def label_encode(self, cols: List[str], save_model: bool = True) -> "Encoder":
        label_encoder = LabelEncoder()
        for col in cols:
            self.df[col] = label_encoder.fit_transform(self.df[col])
        if save_model:
            self.save_with_pickle(label_encoder, "./models/3Label_encoder.pkl")
        return self
    
    def get_dataframe(self) -> pd.DataFrame:
        return self.df
    
class ModelHandler(SaveModel):
    
    def __init__(self, model: XGBRegressor):
        self.model = model
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, param_grid: Dict[str, List[Union[int, float]]]) -> XGBRegressor:
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )

        print("Training model with GridSearchCV...")
        grid_search.fit(X_train, y_train)
        
        print(f'Best params: {grid_search.best_params_}')
        print(f'Best score: {grid_search.best_score_}')
        self.model = grid_search.best_estimator_
        
        return self.model
        
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        predictions = self.model.predict(X_test)
        print(f'Mean Squared Error: {mean_squared_error(y_test, predictions)}')
        print(f'Mean Absolute Error: {mean_absolute_error(y_test, predictions)}')
        print(f'Root Mean Squared Error: {root_mean_squared_error(y_test, predictions)}')
        print(f'R2 Score: {r2_score(y_test, predictions)}')
        
    @staticmethod
    def load_model(file_path: str) -> XGBRegressor:
        with open(file_path, "rb") as f:
            return pickle.load(file_path)   

class PipelineManager:
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def process_data(self, cols_to_drop: List[str], categorica_cols_ohe: List[str], categorica_cols_le: List[str], outliers_vals: dict, cols_for_convert_to_int: List[str], cols_for_convert_to_float: List[str], cols_to_int: List[str]) -> pd.DataFrame:
        processor = DataProcessor(self.df)
        processor.drop_unnecessary_and_NA_values(cols_to_drop)\
            .remove_dash_symbol()\
            .remove_POA_values()\
            .from_cat_to_int(cols_for_convert_to_int)\
            .from_cat_to_float(cols_for_convert_to_float)\
            .convert_to_int(cols_to_int)
        self.df = processor.get_dataframe()
        
        encoder = Encoder(self.df)
        encoder.one_hot_encode(categorica_cols_ohe)
        encoder.label_encode(categorica_cols_le)
        self.df = encoder.get_dataframe()
        
        return self.df
    

if __name__ == "__main__":
    
    data = pd.read_csv("./data/vehical.csv")
    
    pipeline = PipelineManager(data)
    processed_data = pipeline.process_data(
        cols_to_drop=['Title', "Model", "Car/Suv", "Location", "Engine", "ColourExtInt"],
        categorica_cols_ohe=["UsedOrNew", "Transmission", "DriveType", "FuelType"],
        categorica_cols_le=['Brand', "BodyType"],
        cols_for_convert_to_int=['Seats', 'Doors', 'CylindersinEngine', 'Kilometres'],
        cols_for_convert_to_float=['FuelConsumption'],
        cols_to_int=['Price', 'Year'],
        outliers_vals={"Year": (2000, 2024),"FuelConsumption": (1.0, 25.0), "CylindersinEngine": (2, 10), "Seats": (2, 15), "Price": (1000, 100000)}
    )

    X = processed_data.drop(columns=["Price"])
    y = processed_data["Price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model_handler = ModelHandler(XGBRegressor(n_estimators=1000, objective='reg:squarederror', random_state=1234))
    param_grid = {
    'n_estimators': [500, 1000],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.1, 0.2],
    }
    
    model = model_handler.train(X_train, y_train, param_grid)
    model_handler.evaluate(X_test, y_test)

    model_handler.save_with_pickle(model, "./models/3xgb_model.pkl")
