from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class ModelEvaluator:
    def evaluate(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print("Model Evaluation:")
        print(f"MSE : {mse:.2f}")
        print(f"MAE : {mae:.2f}")
        print(f"R²  : {r2:.4f}")

        return {"mse": mse, "mae": mae, "r2": r2}
