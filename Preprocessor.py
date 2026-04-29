#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project: Customer Churn Prediction
Author: Ameesha
Created on: Mon Jul 28 2025
Description:
    Preprocessing class for Customer Churn Prediction project.
    Handles:
        - Missing values
        - Scaling numeric features
        - One-hot encoding categorical features
        - Train-test splitting
        - Transforming new data for inference
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from config import TARGET_COLUMN, TEST_SIZE, RANDOM_STATE

class Preprocessor:
    def __init__(self, drop_columns=None):
        """
        Parameters:
            drop_columns (list): List of columns to drop before processing (e.g., IDs)
        """
        self.target_column = TARGET_COLUMN
        self.column_transformer = None
        self.feature_names_out = None
        self.drop_columns = drop_columns if drop_columns else []

    def split_and_transform(self, df):
        """
        Splits the dataframe into train/test sets and applies preprocessing:
        - Numeric: imputation + scaling
        - Categorical: imputation + one-hot encoding

        Returns:
            X_train_transformed, X_test_transformed, y_train, y_test
        """
        # Drop unwanted columns if any
        df = df.drop(columns=self.drop_columns, errors='ignore')

        # Separate target
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        # Auto-detect column types
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # Save for external transformation
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features

        # Pipelines
        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        self.column_transformer = ColumnTransformer([
            ('num', numeric_pipeline, numeric_features),
            ('cat', categorical_pipeline, categorical_features)
        ])

        # Stratified train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

        # Fit & transform
        X_train_transformed = self.column_transformer.fit_transform(X_train)
        X_test_transformed = self.column_transformer.transform(X_test)

        # Save final feature names
        self.feature_names_out = self.get_feature_names()

        return X_train_transformed, X_test_transformed, y_train, y_test

    def transform_new_data(self, input_data):
        """
        Transform new incoming data (for inference) into the same format
        as training data.
        Accepts either a dict {feature: value} or a DataFrame.
        """
        # If dict, convert to DataFrame
        if isinstance(input_data, dict):
            input_df = pd.DataFrame({k: [v] for k, v in input_data.items()})
        elif isinstance(input_data, pd.DataFrame):
            input_df = input_data.copy()
        else:
            raise ValueError("Input must be a dict or pandas DataFrame")

        # Ensure all training columns exist
        missing_cols = set(self.numeric_features + self.categorical_features) - set(input_df.columns)
        for col in missing_cols:
            input_df[col] = 0  # default value for missing columns

        # Keep only expected columns in correct order
        input_df = input_df[self.numeric_features + self.categorical_features]

        return self.column_transformer.transform(input_df)

    def get_feature_names(self):
        """
        Returns the final feature names after preprocessing (useful for inspection)
        """
        num = self.numeric_features
        cat = self.column_transformer.named_transformers_['cat'].get_feature_names_out(self.categorical_features)
        return list(num) + list(cat)
