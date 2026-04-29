import pandas as pd

class Predictor:
    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor

    def predict(self, input_data_dict):
        """
        Predict for a single-row input.
        input_data_dict: dict {feature_name: value}
        """
        # Transform using preprocessor (it handles dict → DataFrame internally)
        X_transformed = self.preprocessor.transform_new_data(input_data_dict)

        # Make prediction
        prediction = self.model.predict(X_transformed)

        return prediction.tolist()
