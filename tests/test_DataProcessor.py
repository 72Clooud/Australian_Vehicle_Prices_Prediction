import pytest
import pandas as pd

from scripts.train import DataProcessor

class TestDataProcessor:
    
    @pytest.fixture
    def setup(self):
        data = {
        "Brand": ["Toyota", "Honda", "BMW", "Toyota"],
        "Model": ["Corolla", "Civic", "X3", "-"],
        "Year": [2015, 2018, 2020, 2017],
        "UsedOrNew": ["Used", "New", "Used", "New"],
        "Transmission": ["Automatic", "Manual", "Automatic", "Automatic"],
        "DriveType": ["FWD", "AWD", "FWD", "AWD"],
        "FuelType": ["Petrol", "Diesel", "Petrol", "Diesel"],
        "FuelConsumption": ["5.2 L / 100 km", "7.1 L / 100 km", "10.3 L / 100 km", "8.9 L / 100 km"],
        "Kilometres": [80000, 50000, 30000, 70000],
        "CylindersinEngine": [4, 6, 4, 6],
        "BodyType": ["Sedan", "SUV", "Sedan", "SUV"],
        "Doors": [4, 5, 4, 5],
        "Seats": ["5 Seats", "7 Seats", "10 Seats", "8 Seats"],
        "Price": ["20000", "15000", "30000", "40000"]
        }
        df = pd.DataFrame(data)
        data_processor = DataProcessor(df)
        return data_processor
    
    def test_drop_unnecessary_and_NA_values(self, setup): 
        data_processor = setup
        data_processor.df.loc[0, 'Price'] = None    
        cols_to_drop = ['Price', 'FuelConsumption']
        result = data_processor.drop_unnecessary_and_NA_values(cols_to_drop).get_dataframe()
        
        for col in cols_to_drop:
            assert col not in result.columns, f"Column {col} was not removed."
        
        assert result.isnull().sum().sum() == 0, "There are still NaN values in the DataFrame."
        assert 'Brand' in result.columns, "Column 'Brand' should remain."
        assert 'Year' in result.columns, "Column 'Year' should remain."

    def test_remove_dash_symbol(self, setup):
        
        data_processor = setup
        result = data_processor.remove_dash_symbol().get_dataframe()
        assert len(result) == 3, "The number of rows after removing dash symbols is incorrect."
        result_no_dash = data_processor.remove_dash_symbol().get_dataframe()
        assert len(result_no_dash) == 3, "Rows should not be removed when no dash symbols are present."
        
    def test_remove_POA_values(self, setup):
        
        data_processor = setup
        data_processor.df.loc[1, 'Price'] = "POA" 
        result = data_processor.remove_POA_values().get_dataframe()
        assert not (result['Price'] == 'POA').any(), "There are still 'POA' values in the Price column."
        assert len(result) == 3, "The number of rows after removing 'POA' values is incorrect."
        result_no_poa = data_processor.remove_POA_values().get_dataframe()
        assert len(result_no_poa) == 3, "Rows should not be removed when no 'POA' values are present."

    def test_from_cat_to_int(self, setup):
        
        data_processor = setup
        data_processor.from_cat_to_int(['Seats'])
        assert data_processor.df["Seats"].dtype == "int64", "The 'Seats' column was not converted to integers."
        assert data_processor.df["Seats"].iloc[0] == 5, "The conversion to integer was not correct."

    def test_from_cat_to_float(self, setup):

        data_processor = setup
        data_processor.from_cat_to_float(["FuelConsumption"])

        assert data_processor.df["FuelConsumption"].dtype == "float64", "The 'FuelConsumption' column was not converted to floats."
        assert data_processor.df["FuelConsumption"].iloc[1] == 7.1, "The conversion to float was not correct."
        

    def test_convert_to_int(self, setup):

        data_processor = setup
        data_processor.convert_to_int(["Year", "Price"])

        assert data_processor.df["Year"].dtype == "int64", "The 'Year' column was not converted to integers."
        assert data_processor.df["Price"].dtype == "int64", "The 'Price' column was not converted to integers."
        assert data_processor.df["Price"].iloc[0] == 20000, "The 'Price' column was not converted correctly."

    def test_remove_outliers(self, setup):
        
        data_processor = setup
        data_processor.df['Year'] = [1995, 2005, 2020, 1980]
        data_processor.df['FuelConsumption'] = [6, 12, 30, 0]
        data_processor.df['CylindersinEngine'] = [4, 2, 16, 8]
        data_processor.df['Seats'] = [5, 7, 4, 20]
        
        result = data_processor.remove_outliers({"Year": (1994, 2024), "FuelConsumption": (1, 15), "CylindersinEngine": (1, 17), "Seats": (2, 7)}).get_dataframe()

        assert len(result) == 2, "The number of rows after removing outliers is incorrect."
        assert result["Year"].iloc[0] == 1995, "The 'Year' column filter did not work correctly."
        assert result["FuelConsumption"].iloc[1] == 12, "The 'FuelConsumption' column filter did not work correctly."
        assert result["CylindersinEngine"].iloc[0] == 4, "The 'CylindersinEngine' column filter did not work correctly."
        assert result["Seats"].iloc[1] == 7, "The 'Seats' column filter did not work correctly."
