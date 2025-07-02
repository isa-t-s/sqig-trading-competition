import pandas as pd
import numpy as np
import os
from typing import Optional, Tuple


class DataLoader:
    """Class to handle data loading operations."""
    
    def __init__(self, data_path: str):
        """
        Args:
            data_path (str): path to data file
        """
        self.data_path = data_path
    
    def load_data(self, parse_dates: bool = True) -> pd.DataFrame:
        """
        Load the data from a CSV file.

        Args:
            parse_dates (bool, optional): Whether to parse the date column. Defaults to True.
        Returns:
            pd.DataFrame: The loaded data as a DataFrame.
        """
        try:
            if parse_dates:
                df = pd.read_csv(self.data_path, index_col=0, parse_dates=True)
            else:
                df = pd.read_csv(self.data_path, index_col=0)
            
            print(f"Data loaded from {self.data_path}")
            print(f"Shape: {df.shape}")
            print(f"Date range: {df.index.min()} to {df.index.max()}")
            
            return df
       
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
        except Exception as e:
            raise Exception(f"Error loading data from {self.data_path}: {e}")
       
    def get_data_info(self, df: pd.DataFrame) -> dict:
        """
        Get basic information about the dataset.

        Args:
            df: DataFrame to analyze
        Returns:
            Dictionary with data information
        """
        info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'date_range': (df.index.min(), df.index.max()) if hasattr(df.index, 'min') else None,
            'memory_usage': df.memory_usage(deep=True).sum(),
            'duplicates': df.duplicated().sum()
        }
        return info
   
    def print_data_summary(self, df: pd.DataFrame):
        """
        Print a summary of the dataset.
        """
        info = self.get_data_info(df)
        
        print("=" * 50)
        print("DATA SUMMARY")
        print("=" * 50)
        print(f"Shape: {info['shape']}")
        print(f"Columns: {info['columns']}")
        print(f"Memory usage: {info['memory_usage'] / 1024:.2f} KB")
        print(f"Duplicate rows: {info['duplicates']}")
        
        if info['date_range']:
            print(f"Date range: {info['date_range'][0]} to {info['date_range'][1]}")
       
        print("\nMissing values:")
        total_missing = sum(info['missing_values'].values())
        if total_missing == 0:
            print("  No missing values found")
        else:
            for col, missing in info['missing_values'].items():
                if missing > 0:
                    print(f"  {col}: {missing}")
       
        print("\nData types:")
        for col, dtype in info['dtypes'].items():
            print(f"  {col}: {dtype}")
       
        print("\nFirst few rows:")
        print(df.head())
       
        print("\nBasic statistics:")
        print(df.describe())


def load_financial_data(file_path: str) -> pd.DataFrame:
    """
    Helper function to load financial data.

    Args:
        file_path: Path to data file
    Returns:
        DataFrame with financial data
    """
    loader = DataLoader(file_path)
    return loader.load_data()


if __name__ == "__main__":
    # Test the data loader
    try:
        file = "data/raw/sqig_credit_spread_volatility.csv"
        loader = DataLoader(file)
        df = loader.load_data()
        loader.print_data_summary(df)
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your data file exists at the specified path.")