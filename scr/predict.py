import pandas as pd
import numpy as np
from utils import load_model

def make_predictions(model, data):
    """
    Gera previsões e probabilidades com um modelo treinado
    """
    probabilities = model.predict_proba(data)[:,1]
    predictions = (probabilities > 0.40).astype(int)
    
    return predictions, probabilities

if __name__=="__main__":
    # Carregar os dados de teste
    data = pd.read_csv("data/processed/test.csv")

    # Carregar o modelo treinado
    model = load_model("models/classifier.pkl")

    # Fazer previsões
    predictions, probabilities = make_predictions(model, data)

    # Adicionar previsões ao DataFrame
    data['predicted'] = predictions
    data['pred_probability'] = probabilities

    # Selecionar apenas as colunas necessárias
    data = data[['Churn', 'predicted', 'pred_probability']]

    # Salvar predições em um arquivo .xlsx
    data.to_excel("data/processed/predictions.xlsx", index=False)
    print(f'\nPredições salvas em "data/processed/predictions.xlsx"')