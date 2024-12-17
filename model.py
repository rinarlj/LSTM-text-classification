from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
import numpy as np
from tensorflow.keras.layers import Embedding,Flatten, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

    
class LSTMClassifier:
    def __init__(self, vocab_size=10000, max_length=200, embedding_dim=50, lstm_units=128):
        """
        Initialisation du modèle LSTM pour la classification.

        Args:
        - vocab_size : Taille maximale du vocabulaire pour le tokenizer.
        - max_length : Longueur maximale des séquences après padding.
        - embedding_dim : Dimension des vecteurs d'embedding.
        - lstm_units : Nombre d'unités dans la couche LSTM.
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units

    def preprocess(self, texts, labels):
        """
        Prétraitement des données textuelles : Tokenization et Padding.

        Args:
        - texts : Liste des critiques textuelles.
        - labels : Liste des étiquettes associées (0 ou 1).

        Returns:
        - X_padded : Séquences tokenisées et remplies.
        - y : Labels sous forme de tableau numpy.
        """
        tokenizer = Tokenizer(num_words=self.vocab_size)
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        X_padded = pad_sequences(sequences, maxlen=self.max_length, padding='post')
        y = np.array(labels)
        self.tokenizer = tokenizer
        return X_padded, y

    def build_model(self, bidirectional=False):
        """
        Construction du modèle LSTM.

        Args:
        - bidirectional : Si True, ajoute une couche LSTM bidirectionnelle.

        Returns:
        - Modèle Keras compilé.
        """
        model = Sequential()
        model.add(Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, input_length=self.max_length))
        if bidirectional:
            model.add(Bidirectional(LSTM(self.lstm_units)))  # Supprime return_sequences si non nécessaire
        else:
            model.add(LSTM(self.lstm_units))
        model.add(Dropout(0.5))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))  # Sortie binaire
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model = model
        return model


    def train(self, X_train, y_train, X_val, y_val, batch_size=128, epochs=10):
        """
        Entraînement du modèle.

        Args:
        - X_train : Données d'entraînement.
        - y_train : Labels d'entraînement.
        - X_val : Données de validation.
        - y_val : Labels de validation.
        - batch_size : Taille du batch pour l'entraînement.
        - epochs : Nombre d'époques d'entraînement.
        """
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            verbose=1
        )

    def evaluate(self, X_test, y_test):
        """
        Évaluation du modèle sur les données de test.

        Args:
        - X_test : Données de test.
        - y_test : Labels de test.

        Returns:
        - Précision sur les données de test.
        """
        y_pred = (self.model.predict(X_test) > 0.5).astype("int32")
        return accuracy_score(y_test, y_pred)