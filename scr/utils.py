import joblib
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(path):
    """
    Carrega os dados e faz os tratamentos necessários
    """
    df = pd.read_csv(path)

    # Corrigindo o mapeamento de SeniorCitizen
    df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 0: 'Yes'})

    # Convertendo a variável target para binária
    df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

    # Tratando valores faltantes e problemas de preenchimento
    df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
    df['TotalCharges'] = df['TotalCharges'].astype(float)

    return df

def split_data(X, y, test_size = 0.20, random_state = 42):
    """
    Divide os dados em treino e teste
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def load_model(path):
    """
    Carrega o modelo treinado
    """
    return joblib.load(path)