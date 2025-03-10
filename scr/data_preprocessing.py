import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from feature_engine.imputation import MeanMedianImputer, CategoricalImputer
from category_encoders import TargetEncoder

def get_preprocessor():
    """
    Retorna um pré-processador para transformar variáveis categóricas e numéricas

    - Variáveis categóricas: Imputação pelo valor mais frequente e encoding por Target Encoding
    - Variáveis numéricas: Imputação pela mediana
    """
    cat_features = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
        'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]
    num_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

    # Pipeline para variáveis categóricas
    cat_transformer = Pipeline([
        ('cat_imput', CategoricalImputer(imputation_method = 'frequent')),
        ('cat_encoding', TargetEncoder())
    ])

    # Pipeline para variáveis numéricas
    num_transformer = Pipeline([
        ('num_imput', MeanMedianImputer(imputation_method = 'median'))
    ])

    # Aplicando os transformadores
    preprocessor = ColumnTransformer(
        transformers = [
            ('cat', cat_transformer, cat_features),
            ('num', num_transformer, num_features)
        ],
        remainder = 'passthrough'
    )

    return preprocessor