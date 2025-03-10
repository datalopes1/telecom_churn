import joblib
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from data_preprocessing import get_preprocessor
from utils import load_data, split_data

def class_weights(y_train):
    """
    Calcula os pesos das classes para balancear o modelo
    """
    class_weights = compute_class_weight(
        "balanced", classes = np.unique(y_train), y = y_train
    )

    weights_dict = {class_label: weight for class_label, weight in zip(np.unique(y_train), class_weights)}

    return weights_dict

def train_model(X_train, y_train, class_weights):
    """
    Treina o modelo CatBoost com os hiperparâmetros otimizados
    """
    params = {'learning_rate': 0.006232617777096432, 
            'depth': 9, 
            'subsample': 0.6803560166453312, 
            'colsample_bylevel': 0.9966486140633338, 
            'min_data_in_leaf': 63}
    
    model = CatBoostClassifier(
        **params, verbose = 0, class_weights=class_weights, random_state=42
    )
    model.fit(X_train, y_train)
    print("\nTreinamento do modelo completo.")
    
    return model

def save_model(model, preprocessor, path):
    """
    Salva o modelo treinado e o pré-processador em um pipeline
    """

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    joblib.dump(pipeline, path)
    print(f"\nModelo salvo em {path}")

if __name__=="__main__":
    # Carregar dados e pré-processador
    data = load_data('data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    preprocessor = get_preprocessor()

    # Definição das features e target
    features = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
        'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
    ]
    target = 'Churn'

    # Dividir os dados em treino e teste
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Calcular pesos das classes
    weights_dict = class_weights(y_train)

    # Salvar conjuntos de treino e teste
    train_data = pd.concat([X_train, y_train], axis = 1)
    train_data.to_csv("data/processed/train.csv", index = False)

    test_data = pd.concat([X_test, y_test], axis = 1)
    test_data.to_csv("data/processed/test.csv", index = False)

    # Transformar os dados de treino
    X_train_transformed = preprocessor.fit_transform(X_train, y_train)
    X_train_transformed = pd.DataFrame(X_train_transformed)

    # Treinar e salar o modelo
    model = train_model(X_train_transformed, y_train, weights_dict)
    save_model(model, preprocessor, "models/classifier.pkl")