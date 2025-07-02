import os

# File paths
DATA_DIR = "data"

# Data file names
RAW_DATA_FILE = "financial_data.csv"
PROCESSED_DATA_FILE = "processed_financial_data.csv"

# Model configuration
FEATURE_COLUMNS = ['T-Bill_13W_Yield', '10Y_Treasury_Yield', 'Credit_Spread']
TARGET_COLUMN = 'VIX'

# Data processing parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
ROLLING_WINDOW = 5

# Visualization settings
FIGURE_SIZE = (15, 10)
STYLE = 'default'