import pytest
import pandas as pd
import os
import pickle


from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from tempfile import NamedTemporaryFile
from io import StringIO

from functions.data_processing import ( 
    save_with_pickle,
    drop_unnecessary_and_NA_values,
    remove_dash_symbol,
    remove_POA_values,
    from_cat_to_num,
    convert_to_int,
    remove_outliers,
    show_unique_values,
    oneHot_encoding,
    label_encoder,
    split_data,
)

data = {
    "Brand": ["Toyota", "Honda", "BMW", "Toyota"],
    "Year": [2015, 2018, 2020, 2017],
    "UsedOrNew": ["Used", "New", "Used", "New"],
    "Transmission": ["Automatic", "Manual", "Automatic", "Automatic"],
    "DriveType": ["FWD", "AWD", "FWD", "AWD"],
    "FuelType": ["Petrol", "Diesel", "Petrol", "Diesel"],
    "FuelConsumption": [7.5, 6.0, 8.0, 5.5],
    "Kilometres": [80000, 50000, 30000, 70000],
    "CylindersinEngine": [4, 6, 4, 6],
    "BodyType": ["Sedan", "SUV", "Sedan", "SUV"],
    "Doors": [4, 5, 4, 5],
    "Seats": [5, 7, 5, 7],
    "Price": ["20000", "POA", "30000", "40000"]
}
df = pd.DataFrame(data)


def test_drop_unnecessary_and_NA_values():
    df_with_na = df.copy()
    df_with_na.loc[0, 'Price'] = None    

    cols_to_drop = ['Price', 'FuelConsumption']
    result = drop_unnecessary_and_NA_values(df_with_na, cols_to_drop)
    
    for col in cols_to_drop:
        assert col not in result.columns, f"Column {col} was not removed."
    
    assert result.isnull().sum().sum() == 0, "There are still NaN values in the DataFrame."
    assert 'Brand' in result.columns, "Column 'Brand' should remain."
    assert 'Year' in result.columns, "Column 'Year' should remain."

def test_remove_dash_symbol():
    data = {
        "Brand": ["Toyota", "Honda", "BMW", "Chevrolet"],
        "Model": ["Corolla", "Civic", "X3", "-"],
        "Year": [2015, 2018, 2020, 2022]
    }
    df = pd.DataFrame(data)
    result = remove_dash_symbol(df)
    assert len(result) == 3, "The number of rows after removing dash symbols is incorrect."
    
    data_no_dash = {
        "Brand": ["Toyota", "Honda", "BMW", "Chevrolet"],
        "Model": ["Corolla", "Civic", "X3", "Silverado"],
        "Year": [2015, 2018, 2020, 2022]
    }
    df_no_dash = pd.DataFrame(data_no_dash)
    result_no_dash = remove_dash_symbol(df_no_dash)
    assert len(result_no_dash) == 4, "Rows should not be removed when no dash symbols are present."
    
def test_remove_POA_values():
    data = {
        "Brand": ["Toyota", "Honda", "BMW", "Chevrolet"],
        "Model": ["Corolla", "Civic", "X3", "S10"],
        "Price": ["20000", "POA", "30000", "POA"]
    }
    df = pd.DataFrame(data)
    result = remove_POA_values(df)
    assert not (result['Price'] == 'POA').any(), "There are still 'POA' values in the Price column."
    assert len(result) == 2, "The number of rows after removing 'POA' values is incorrect."
    
    data_no_poa = {
        "Brand": ["Toyota", "Honda", "BMW", "Chevrolet"],
        "Model": ["Corolla", "Civic", "X3", "S10"],
        "Price": ["20000", "25000", "30000", "35000"]
    }
    df_no_poa = pd.DataFrame(data_no_poa)
    result_no_poa = remove_POA_values(df_no_poa)
    assert len(result_no_poa) == 4, "Rows should not be removed when no 'POA' values are present."

def test_from_cat_to_num():
    data = {
        "Brand": ["Toyota", "Honda", "BMW", "Chevrolet"],
        "Seats": ["5 Seats", "7 Seats", "10 Seats", "8 Seats"]
    }
    df = pd.DataFrame(data)
    from_cat_to_num(df, "Seats", int)
    assert df["Seats"].dtype == "int64", "The 'Seats' column was not converted to integers."
    assert df["Seats"].iloc[0] == 5, "The conversion to integer was not correct."

    data_float = {
        "Brand": ["Toyota", "Honda", "BMW", "Chevrolet"],
        "FuelConsumption": ["5.2 L / 100 km", "7.1 L / 100 km", "10.3 L / 100 km", "8.9 L / 100 km"]
    }
    df_float = pd.DataFrame(data_float)
    from_cat_to_num(df_float, "FuelConsumption", float)

    assert df_float["FuelConsumption"].dtype == "float64", "The 'FuelConsumption' column was not converted to floats."
    assert df_float["FuelConsumption"].iloc[1] == 7.1, "The conversion to float was not correct."
    

def test_convert_to_int():
    data = {
        "Brand": ["Toyota", "Honda", "BMW", "Chevrolet"],
        "Year": [2015, 2018, 2020, 2022],
        "Price": ["20000", "15000", "30000", "40000"]
    }

    df = pd.DataFrame(data)
    df["Price"] = df["Price"].astype(str)

    df = convert_to_int(df, ["Year", "Price"])

    assert df["Year"].dtype == "int64", "The 'Year' column was not converted to integers."
    assert df["Price"].dtype == "int64", "The 'Price' column was not converted to integers."
    assert df["Price"].iloc[0] == 20000, "The 'Price' column was not converted correctly."

def test_remove_outliers():
    data = {
        "Year": [1995, 2005, 2020, 1980],
        "FuelConsumption": [6, 12, 30, 0],
        "CylindersinEngine": [4, 2, 16, 8],
        "Seats": [5, 7, 4, 20]
    }
    df = pd.DataFrame(data)
    cols = ["Year", "FuelConsumption", "CylindersinEngine", "Seats"]

    result = remove_outliers(df, cols)

    assert len(result) == 2, "The number of rows after removing outliers is incorrect."
    assert result["Year"].iloc[0] == 1995, "The 'Year' column filter did not work correctly."
    assert result["FuelConsumption"].iloc[1] == 12, "The 'FuelConsumption' column filter did not work correctly."
    assert result["CylindersinEngine"].iloc[0] == 4, "The 'CylindersinEngine' column filter did not work correctly."
    assert result["Seats"].iloc[1] == 7, "The 'Seats' column filter did not work correctly."

    data_invalid = {
        "Year": [1995, 2005, 2020, 1980],
        "FuelConsumption": [6, -12, 30, 0],
        "CylindersinEngine": [4, 2, -16, 8],
        "Seats": [5, -7, 4, 20]
    }
    df_invalid = pd.DataFrame(data_invalid)
    result_invalid = remove_outliers(df_invalid, cols)

    assert result_invalid.shape[0] == 1, "The number of rows after filtering outliers is incorrect in the invalid data."

def test_show_unique_values(capfd):
    data = {
        "Brand": ["Toyota", "Honda", "BMW", "Toyota", "Honda"],
        "Model": ["Corolla", "Civic", "X3", "Corolla", "Civic"],
        "Year": [2015, 2018, 2020, 2015, 2018]
    }
    df = pd.DataFrame(data)
    cols = ["Brand", "Model", "Year"]

    show_unique_values(df, cols)
    captured = capfd.readouterr()
    output = captured.out.strip().split("\n")
    
    assert output == ['3', '3', '3'], f"Expected unique values counts ['3', '3', '3'], but got {output}."
    
    data_no_duplicates = {
        "Brand": ["Toyota", "Honda", "BMW"],
        "Model": ["Corolla", "Civic", "X3"],
        "Year": [2015, 2018, 2020]
    }
    df_no_duplicates = pd.DataFrame(data_no_duplicates)
    show_unique_values(df_no_duplicates, cols)

    captured_no_duplicates = capfd.readouterr()
    output_no_duplicates = captured_no_duplicates.out.strip().split("\n")

    assert output_no_duplicates == ['3', '3', '3'], f"Expected unique values counts ['3', '3', '3'], but got {output_no_duplicates}."
    
def test_oneHot_encoding():
    data = {
        "Brand": ["Toyota", "Honda", "BMW"],
        "Model": ["Corolla", "Civic", "X3"],
        "Year": [2015, 2018, 2020]
    }

    df = pd.DataFrame(data)
    categorical_cols = ["Brand", "Model"]

    result = oneHot_encoding(categorical_cols, df)

    assert "Brand_Toyota" in result.columns, "One-hot encoding for 'Brand' column failed."
    assert "Model_Civic" in result.columns, "One-hot encoding for 'Model' column failed."
    assert "Year" in result.columns, "The 'Year' column was dropped incorrectly."
    
    assert result["Brand_Toyota"].iloc[0] == 1, "One-hot encoding value for 'Brand_Toyota' is incorrect."
    assert result["Model_Civic"].iloc[1] == 1, "One-hot encoding value for 'Model_Civic' is incorrect."
    
    assert "Brand" not in result.columns, "The 'Brand' column was not removed after one-hot encoding."
    assert "Model" not in result.columns, "The 'Model' column was not removed after one-hot encoding."

def test_label_encoder():
    data = {
        "Brand": ["Toyota", "Honda", "BMW", "Toyota", "Honda"],
        "Model": ["Corolla", "Civic", "X3", "Corolla", "Civic"],
        "Year": [2015, 2018, 2020, 2015, 2018]
    }

    df = pd.DataFrame(data)
    categorical_cols = ["Brand", "Model"]

    result = label_encoder(categorical_cols, df)

    assert result["Brand"].iloc[0] == 2, "Label encoding for 'Brand' column failed."
    assert result["Model"].iloc[1] == 0, "Label encoding for 'Model' column failed."
    
    assert result["Brand"].nunique() == 3, "The 'Brand' column does not have the correct number of unique labels."
    assert result["Model"].nunique() == 3, "The 'Model' column does not have the correct number of unique labels."

    assert result["Brand"].iloc[2] != result["Brand"].iloc[0], "Label encoding did not assign unique integers to 'Brand'."
    assert result["Model"].iloc[4] != result["Model"].iloc[3], "Label encoding did not assign unique integers to 'Model'."

def test_split_data():
    data = {
        "Brand": ["Toyota", "Honda", "BMW", "Toyota", "Honda"],
        "Model": ["Corolla", "Civic", "X3", "Corolla", "Civic"],
        "Year": [2015, 2018, 2020, 2015, 2018],
        "Price": [20000, 25000, 30000, 22000, 26000]
    }

    df = pd.DataFrame(data)
    target = "Price"
    test_size = 0.4

    X_train, X_test, y_train, y_test = split_data(df, target, test_size)

    assert len(X_train) == 3, f"Expected 3 rows in X_train, but got {len(X_train)}."
    assert len(X_test) == 2, f"Expected 2 rows in X_test, but got {len(X_test)}."
    
    assert len(y_train) == 3, f"Expected 3 rows in y_train, but got {len(y_train)}."
    assert len(y_test) == 2, f"Expected 2 rows in y_test, but got {len(y_test)}."

    assert target not in X_train.columns, f"Target column '{target}' found in X_train."
    assert target not in X_test.columns, f"Target column '{target}' found in X_test."
    
    assert target in y_train.name, f"Target column '{target}' not found in y_train."
    assert target in y_test.name, f"Target column '{target}' not found in y_test."