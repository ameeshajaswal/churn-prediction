from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

class ModelTrainer:
    def __init__(self, model_type="linear"):
        if model_type.lower() == "linear":
            self.model = LinearRegression()
        elif model_type.lower() == "random_forest":
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError("Supported models: linear, random_forest")

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def get_model(self):
        return self.model
