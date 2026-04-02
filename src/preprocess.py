import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

def load_and_preprocess(csv_path: str, save_scaler: bool = True):
    df = pd.read_csv(csv_path)

    # Audio features expected from Spotify API or similar
    feature_cols = [
        "danceability", "energy", "loudness", "speechiness",
        "acousticness", "instrumentalness", "liveness",
        "valence", "tempo", "duration_ms"
    ]

    df = df.dropna(subset=feature_cols)
    df = df.reset_index(drop=True)

    X = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if save_scaler:
        joblib.dump(scaler, "models/scaler.pkl")

    return df, X_scaled, feature_cols