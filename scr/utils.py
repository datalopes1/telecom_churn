import joblib
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(path):
    df = pd.read_csv(path)
    df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 0: 'Yes'})
    df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})
    df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
    df['TotalCharges'] = df['TotalCharges'].astype(float)

    return df

def split_data(X, y, test_size = 0.20, random_state = 42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def load_model(path):
    return joblib.load(path)