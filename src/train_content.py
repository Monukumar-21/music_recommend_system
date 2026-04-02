import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.neighbors import NearestNeighbors
import joblib
from preprocess import load_and_preprocess

# --- Autoencoder ---
def build_autoencoder(input_dim: int, latent_dim: int = 32):
    # Encoder
    inputs = keras.Input(shape=(input_dim,), name="encoder_input")
    x = layers.Dense(64, activation="relu")(inputs)
    z = layers.Dense(latent_dim, activation="relu", name="latent")(x)

    # Decoder
    x = layers.Dense(64, activation="relu")(z)
    outputs = layers.Dense(input_dim, name="decoder_output")(x)

    autoencoder = keras.Model(inputs, outputs, name="autoencoder")
    encoder     = keras.Model(inputs, z, name="encoder")

    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder, encoder


def train_autoencoder(X_scaled: np.ndarray, epochs: int = 50, batch_size: int = 64):
    input_dim = X_scaled.shape[1]
    autoencoder, encoder = build_autoencoder(input_dim, latent_dim=32)

    autoencoder.fit(
        X_scaled, X_scaled,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_split=0.1,
        verbose=1
    )

    # Save in TensorFlow SavedModel format
    autoencoder.save("models/autoencoder")
    encoder.save("models/encoder")
    return autoencoder, encoder


def build_knn(encoder: keras.Model, X_scaled: np.ndarray, n_neighbors: int = 10):
    embeddings = encoder.predict(X_scaled, verbose=0)

    knn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    knn.fit(embeddings)

    joblib.dump(knn, "models/knn_model.pkl")
    joblib.dump(embeddings, "models/embeddings.pkl")
    return knn, embeddings


if __name__ == "__main__":
    df, X_scaled, _ = load_and_preprocess("data/tracks.csv")
    autoencoder, encoder = train_autoencoder(X_scaled)
    build_knn(encoder, X_scaled)
    print("Content-based model saved.")