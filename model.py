import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
import numpy as np
import re
from collections import Counter

# --- Classe LSTMClassifier ---
class LSTMClassifier:
    def __init__(self, vocab_size=10000, max_length=200, embedding_dim=50, lstm_units=128):
        """
        Initialisation du modèle LSTM pour la classification.

        Args:
        - vocab_size : Taille maximale du vocabulaire.
        - max_length : Longueur maximale des séquences après padding.
        - embedding_dim : Dimension des vecteurs d'embedding.
        - lstm_units : Nombre d'unités dans la couche LSTM.
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.tokenizer = None
        self.model = None

    # --- Prétraitement des données ---
    def preprocess(self, texts, labels):
        """
        Tokenisation et padding des textes.

        Args:
        - texts : Liste des textes.
        - labels : Liste des étiquettes.

        Returns:
        - X_padded : Séquences tokenisées et remplies.
        - y : Labels sous forme de tenseurs.
        """
        # Tokenizer simple pour PyTorch
        all_words = [word for text in texts for word in re.findall(r'\b\w+\b', text.lower())]
        counter = Counter(all_words)
        vocab = {word: idx + 2 for idx, (word, _) in enumerate(counter.most_common(self.vocab_size))}
        vocab['<PAD>'] = 0
        vocab['<UNK>'] = 1

        self.tokenizer = vocab  # Sauvegarde du vocabulaire

        def encode(text):
            tokens = [vocab.get(word, vocab['<UNK>']) for word in re.findall(r'\b\w+\b', text.lower())]
            if len(tokens) < self.max_length:
                tokens += [0] * (self.max_length - len(tokens))  # Padding
            return tokens[:self.max_length]

        X_padded = torch.tensor([encode(text) for text in texts], dtype=torch.long)
        y = torch.tensor(labels, dtype=torch.float32)
        return X_padded, y

    # --- Construction du modèle LSTM ---
    def build_model(self, bidirectional=False):
        """
        Construction du modèle LSTM.

        Args:
        - bidirectional : Si True, utilise une LSTM bidirectionnelle.
        """
        class LSTMModel(nn.Module):
            def __init__(self, vocab_size, embedding_dim, lstm_units, max_length, bidirectional):
                super(LSTMModel, self).__init__()
                self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
                self.lstm = nn.LSTM(embedding_dim, lstm_units, batch_first=True, bidirectional=bidirectional)
                self.dropout = nn.Dropout(0.5)
                self.fc1 = nn.Linear(lstm_units * (2 if bidirectional else 1), 128)
                self.fc2 = nn.Linear(128, 1)
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                x = self.embedding(x)
                _, (hidden, _) = self.lstm(x)
                x = hidden[-1] if not bidirectional else torch.cat((hidden[-2], hidden[-1]), dim=1)
                x = self.dropout(torch.relu(self.fc1(x)))
                return self.sigmoid(self.fc2(x)).squeeze()

        self.model = LSTMModel(self.vocab_size, self.embedding_dim, self.lstm_units, self.max_length, bidirectional)
        print("Modèle LSTM créé.")

    # --- Entraînement ---
    def train(self, X_train, y_train, X_val, y_val, batch_size=128, epochs=10, lr=0.001):
        """
        Entraînement du modèle.

        Args:
        - X_train, y_train : Données d'entraînement.
        - X_val, y_val : Données de validation.
        """
        train_data = torch.utils.data.TensorDataset(X_train, y_train)
        val_data = torch.utils.data.TensorDataset(X_val, y_val)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # Évaluation sur les données de validation
            self.model.eval()
            val_preds, val_targets = [], []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = self.model(X_batch)
                    val_preds.extend((outputs > 0.5).cpu().numpy())
                    val_targets.extend(y_batch.cpu().numpy())
            val_acc = accuracy_score(val_targets, val_preds)
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f} - Val Accuracy: {val_acc:.4f}")

    # --- Évaluation ---
    def evaluate(self, X_test, y_test):
        """
        Évaluation du modèle sur les données de test.

        Args:
        - X_test : Données de test.
        - y_test : Étiquettes de test.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        self.model = self.model.to(device)

        X_test, y_test = X_test.to(device), y_test.to(device)
        with torch.no_grad():
            outputs = self.model(X_test)
            preds = (outputs > 0.5).cpu().numpy()
        acc = accuracy_score(y_test.cpu().numpy(), preds)
        print(f"Test Accuracy: {acc:.4f}")
        return acc
