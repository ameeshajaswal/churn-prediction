# -*- coding: utf-8 -*-
"""
Project: Customer Churn Prediction
Author: Ameesha
Created on: Mon Jul 28 2025
Description: 
    Configuration file containing dataset paths, target column, 
    test/train split ratio, random state, and optional feature lists
    for the Customer Churn Prediction project.
"""
import pandas as pd

class DataLoader:
    def __init__(self, path):
        self.path = path

    def load_data(self):
        df = pd.read_csv(self.path)
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
