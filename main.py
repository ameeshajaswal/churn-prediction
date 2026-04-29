from DataLoader import DataLoader
from Preprocessor import Preprocessor
from ModelTrainer import ModelTrainer
from ModelEvaluator import ModelEvaluator
from Predictor import Predictor
from config import DATA_PATH, TARGET_COLUMN

def main():
    # 1. Load data
    loader = DataLoader(DATA_PATH)
    df = loader.load_data()
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Strip column names to avoid hidden spaces
    df.columns = df.columns.str.strip()

    # 2. Convert target column to numeric if needed
    if df[TARGET_COLUMN].dtype == object:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].map({'Yes': 1, 'No': 0})

    # 3. Preprocess
    preprocessor = Preprocessor()
    X_train, X_test, y_train, y_test = preprocessor.split_and_transform(df)

    # 4. Train models
    trainer_lr = ModelTrainer("linear")
    trainer_lr.train(X_train, y_train)
    model_lr = trainer_lr.get_model()

    trainer_rf = ModelTrainer("random_forest")
    trainer_rf.train(X_train, y_train)
    model_rf = trainer_rf.get_model()

    # 5. Evaluate models
    evaluator = ModelEvaluator()
    print("\n--- Linear Regression ---")
    evaluator.evaluate(model_lr, X_test, y_test)
    print("\n--- Random Forest ---")
    evaluator.evaluate(model_rf, X_test, y_test)

    # 6. Predict example
    # Make sure keys exactly match features used in training
    sample_input = {
        "Incoming Calls": 100,
        "Answered Calls": 95,
        "Answer Rate": 0.95,
        "Abandoned Calls": 5,
        "Answer Speed (AVG)": 12,
        "Talk Duration (AVG)": 180,
        "Waiting Time (AVG)": 15,
        "Service Level (20 Seconds)": 0.8,
        "Status": 0  # Added: 0 = No, 1 = Yes
    }

    predictor = Predictor(model_rf, preprocessor)
    prediction = predictor.predict(sample_input)
    print(f"\nPredicted Call Volume / Churn Probability: {prediction[0]:.2f}")

if __name__ == "__main__":
    main()
