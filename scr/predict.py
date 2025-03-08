import pandas as pd
import numpy as np
from utils import load_model

def make_predictions(model, data):
    probabilities = model.predict_proba(data)[:,1]
    predictions = (probabilities > 0.40).astype(int)
    
    return predictions, probabilities

if __name__=="__main__":

    data = pd.read_csv("data/processed/test.csv")
    model = load_model("models/classifier.pkl")

    predictions, probabilities = make_predictions(model, data)
    data['predicted'] = predictions
    data['pred_probability'] = probabilities

    data = data[['Churn', 'predicted', 'pred_probability']]
    data.to_excel("data/processed/predictions.xlsx", index = False)
    print(f'\nPredições salvas em "data/processed/predictions.xlsx"')