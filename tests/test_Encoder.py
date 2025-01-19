import pytest
import pandas as pd
from scripts.train import Encoder

class TestEncoder:
    
    @pytest.fixture()
    def setup(self):
        data = {
            "Brand": ["Toyota", "Honda", "BMW"],
            "Model": ["Corolla", "Civic", "X3"],
            "UsedOrNew": ["USED", "USED", "NEW"],
            "BodyType": ["SUV", "Wagon", "Wagon"],
            "Year": [2015, 2018, 2020]
        }
        df = pd.DataFrame(data)
        encoder = Encoder(df)
        return encoder
    
    def test_one_hot_encode(self, setup):
        encoder = setup
        categorical_cols = ["Brand", "Model"]
        result = encoder.one_hot_encode(categorical_cols, save_model=False).get_dataframe()

        assert "Brand_Toyota" in result.columns, "One-hot encoding for 'Brand' column failed."
        assert "Model_Civic" in result.columns, "One-hot encoding for 'Model' column failed."
        assert "Year" in result.columns, "The 'Year' column was dropped incorrectly."
        
        assert result["Brand_Toyota"].iloc[0] == 1, "One-hot encoding value for 'Brand_Toyota' is incorrect."
        assert result["Model_Civic"].iloc[1] == 1, "One-hot encoding value for 'Model_Civic' is incorrect."
        
        assert "Brand" not in result.columns, "The 'Brand' column was not removed after one-hot encoding."
        assert "Model" not in result.columns, "The 'Model' column was not removed after one-hot encoding."
        
    def test_label_encoder(self, setup): 
        encoder = setup
        categorical_cols = ["UsedOrNew", "BodyType"]
        result = encoder.label_encode(categorical_cols, save_model=False).get_dataframe()

        assert result["UsedOrNew"].iloc[0] == 1, "Label encoding for 'UsedOrNew' column failed."
        assert result["BodyType"].iloc[1] == 1, "Label encoding for 'BodyType' column failed."
        
        assert result["UsedOrNew"].nunique() == 2, "The 'UsedOrNew' column does not have the correct number of unique labels."
        assert result["BodyType"].nunique() == 2, "The 'BodyType' column does not have the correct number of unique labels."

        assert result["UsedOrNew"].iloc[0] != result["UsedOrNew"].iloc[2], "Label encoding did not assign unique integers to 'UsedOrNew'."
        assert result["BodyType"].iloc[0] != result["BodyType"].iloc[1], "Label encoding did not assign unique integers to 'BodyType'."
