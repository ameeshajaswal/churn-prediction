# config.py

# Path to the dataset
DATA_PATH = r"E:\Spider\Customer Churn Prediction Project\Call Center Data.csv"

# Target column (what you want to predict)
TARGET_COLUMN = "Status"  # Change this to the column name that indicates if a customer left

# Test split size
TEST_SIZE = 0.2  # 20% of data for testing, 80% for training

# Random seed for reproducibility
RANDOM_STATE = 42

# Optional: columns to drop (like IDs or irrelevant features)
DROP_COLUMNS = ["customerID"]  # adjust if your dataset has an ID column

# Optional: numerical and categorical features (for preprocessing)
NUMERICAL_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]  # example
CATEGORICAL_FEATURES = ["gender", "Contract", "PaymentMethod", "InternetService"]  # example
