import pandas as pd
import numpy as np
from predict import load_model
from utils import split_data, load_data
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score

def evaluation(model, X_test, y_test):
    """
    Avalia o desempenho do modelo no conjunto de teste
    """
    y_proba = model.predict_proba(X_test)[:,1]
    y_pred = model.predict(X_test)

    results = {
        'Acurácia': accuracy_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'ROC AUC': roc_auc_score(y_test, y_proba),
        'MCC': matthews_corrcoef(y_test, y_pred)
    }
    return pd.Series(results)

def cross_validation(model, X, y):
    """
    Executa validação cruzada usando o F1 Score
    """
    scoring = make_scorer(f1_score)
    cv = StratifiedKFold(n_splits = 5)

    scores = cross_val_score(model, X, y, cv = cv, scoring = scoring)
    print("Validação Cruzada")
    print(f"{'-' * 25}")
    print(f"Média dos F1 Scores: {scores.mean()}")
    print(f"Desvio Padráo dos F1 Scores: {scores.std()}")

    return scores.mean(), scores.std(), scores

if __name__=="__main__":
    # Carregar os dados e o modelo treinado
    data = load_data("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    model = load_model("models/classifier.pkl")

    # Separar as features e target
    X = data.drop(columns = ['Churn', 'customerID'], axis = 1)
    y = data['Churn']

    # Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test = split_data(X, y)

    # realizar a validação cruzada
    mean_f1, std_f1, all_scores = cross_validation(model, X_train, y_train)

    # Avaliar o modelo no conjunto de teste
    print("\nMétricas de avaliação")
    print(f"{'-' * 25}")
    print(evaluation(model, X_test, y_test))
   