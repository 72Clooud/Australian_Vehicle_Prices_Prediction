from typing import List, Union, Tuple

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

import pandas as pd
import pickle
import os

def save_with_pickle(model, file_path: str) -> None:
    folder = os.path.dirname("../models")
    if not os.path.exists(folder):
        os.mkdir("../models")
    with open(file_path, "wb") as file:
        pickle.dump(model, file)

def drop_unnecessary_and_NA_values(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    df.drop(columns=cols, inplace=True)
    df.dropna(inplace=True)
    try:
        if df.isnull().sum().sum() == 0:
            print("All NaN values have been removed.")
            return df   
        else:
            print("There are still NaN values in the DataFrame.")
    except KeyError as e:
        raise KeyError(f"One or more columns specified in 'cols' were not found in the DataFrame: {e}")
    
    except ValueError as e:
        raise ValueError(f"A value error occurred, possibly due to mismatched data types: {e}")
    
    except Exception as e:
        raise RuntimeError(f"An error occurred during the operation: {e}")
    
    
def remove_dash_symbol(df: pd.DataFrame) -> pd.DataFrame:
    mask = df.apply(lambda col: col.astype(str).str.contains('-')).any(axis=1)
    df = df[~mask]
    if df.apply(lambda col: col.astype(str).str.contains('-')).any().any():
            raise ValueError("Not all dash symbols ('-') have been removed from the DataFrame.")
    return df

def remove_POA_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df['Price'] != 'POA'].reset_index(drop=True)
    if (df['Price'] == 'POA').any():
        raise ValueError("Not all 'POA' values have been removed from the Price column.")
    return df

def from_cat_to_num(df: pd.DataFrame, col: str, dtypes: Union[int, float]) -> None:
    if dtypes == int:
        df[col] = df[col].str.replace('[^0-9]', '', regex=True).astype(dtypes)
    else:
       df[col] = df[col].str.extract(r'(\d+\.?\d*)').astype(float)
       
def convert_to_int(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for col in cols:
        df[col] = df[col].astype(int)
    return df
    
def remove_outliers(df: pd.DataFrame, cols_to_remove_outliers: List[str]) -> pd.DataFrame:
    df = df.copy()
    df = df[df[cols_to_remove_outliers[0]] > 1990]
    df = df[(df[cols_to_remove_outliers[1]] >= 1) & (df[cols_to_remove_outliers[1]] < 25)]
    df = df[df[cols_to_remove_outliers[2]] > 1]
    df = df[df[cols_to_remove_outliers[3]] < 15]
    df.reset_index(drop=True, inplace=True)
    return df

def show_unique_values(df: pd.DataFrame, cols: List[str]) -> None:
    for col in cols:
        print(len(df[col].unique()))
        
def oneHot_encodeing(categorical_cols: List[str], df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.copy()
        one_hot_encoder = OneHotEncoder(sparse_output=False)
        one_hot_encoded = one_hot_encoder.fit_transform(df[categorical_cols])
        one_hot_df = pd.DataFrame(one_hot_encoded, 
                                  columns=one_hot_encoder.get_feature_names_out(categorical_cols), 
                                  index=df.index) 
        df_encoded = pd.concat([df.drop(categorical_cols, axis=1),
                                one_hot_df], axis=1)
        
        save_with_pickle(one_hot_encoder, "../models/OneHot_encoder.pkl")
        
        return df_encoded
    except KeyError as e:
        raise KeyError(f"One or more columns specified in 'categorical_cols' were not found in the DataFrame: {e}")
    
    except Exception as e:
        raise RuntimeError(f"An error occurred during one-hot encoding: {e}")
    
def label_encoder(categorical_cols: List[str], df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    label_encoder = LabelEncoder()
    df[categorical_cols[0]] = label_encoder.fit_transform(df[categorical_cols[0]])
    df[categorical_cols[1]] = label_encoder.fit_transform(df[categorical_cols[1]])
    
    save_with_pickle(label_encoder, "../models/Label_encoder.pkl")
    return df

def split_data(df: pd.DataFrame, target: str, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df.drop(columns=target)
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    
    return X_train, X_test, y_train, y_test

