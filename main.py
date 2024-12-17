from model import LSTMClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import os

def load_data(csv_path):

    data = pd.read_csv(csv_path)

    texts = data['text'].tolist()
    labels = (data['Label'] == 'bodo').astype(int).tolist()  
    return texts, labels

if __name__ == "__main__":
   
    texts, labels = load_data("data.csv")

    classifier = LSTMClassifier()
    X, y = classifier.preprocess(texts, labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    # --- Étape 5 : Construction et entraînement du modèle ---
    classifier.build_model(bidirectional=True)
    classifier.train(X_train, y_train, X_val, y_val, epochs=5)

    # --- Étape 6 : Évaluation sur les données de test ---
    accuracy = classifier.evaluate(X_test, y_test)
    print(f"Précision finale sur les données de test : {accuracy:.4f}")
