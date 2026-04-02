import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
import joblib

def train_collab(csv_path: str, n_components: int = 20):
    # Expects columns: user_id, track_id, play_count (or rating)
    df = pd.read_csv(csv_path)

    user_enc = {u: i for i, u in enumerate(df["user_id"].unique())}
    track_enc = {t: i for i, t in enumerate(df["track_id"].unique())}

    df["u"] = df["user_id"].map(user_enc)
    df["t"] = df["track_id"].map(track_enc)

    R = csr_matrix(
        (df["play_count"].values, (df["u"].values, df["t"].values))
    ).toarray().astype(float)

    # Mean-center and SVD
    R_mean = R.mean(axis=1, keepdims=True)
    R_demeaned = R - R_mean

    U, sigma, Vt = svds(R_demeaned, k=n_components)
    sigma = np.diag(sigma)

    R_pred = np.dot(np.dot(U, sigma), Vt) + R_mean

    joblib.dump({
        "R_pred": R_pred,
        "user_enc": user_enc,
        "track_enc": track_enc,
        "track_dec": {v: k for k, v in track_enc.items()}
    }, "models/svd_model.pkl")
    print("Collaborative filter saved.")

if __name__ == "__main__":
    train_collab("data/user_plays.csv")