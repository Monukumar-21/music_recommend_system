import numpy as np
import tensorflow as tf
import joblib

class MusicRecommender:
    def __init__(self):
        self.scaler    = joblib.load("models/scaler.pkl")
        self.knn       = joblib.load("models/knn_model.pkl")
        self.embeddings = joblib.load("models/embeddings.pkl")
        self.svd_data  = joblib.load("models/svd_model.pkl")

        # Load the encoder-only model for fast inference
        self.encoder = tf.keras.models.load_model("models/encoder")

    def content_recommend(self, features: list, top_n: int = 10):
        x = np.array(features, dtype=np.float32).reshape(1, -1)
        x_scaled = self.scaler.transform(x)

        # encoder.predict expects a batch
        z = self.encoder.predict(x_scaled, verbose=0)

        distances, indices = self.knn.kneighbors(z)
        return indices[0].tolist(), distances[0].tolist()

    def collab_recommend(self, user_id: str, top_n: int = 10) -> list:
        data = self.svd_data
        if user_id not in data["user_enc"]:
            return []
        u_idx  = data["user_enc"][user_id]
        scores = data["R_pred"][u_idx]
        top_indices = np.argsort(scores)[::-1][:top_n]
        return [data["track_dec"][i] for i in top_indices]

    def hybrid_recommend(self, features: list, user_id: str = None,
                         top_n: int = 10, alpha: float = 0.6):
        content_idx, content_dist = self.content_recommend(features, top_n)
        collab_tracks = self.collab_recommend(user_id, top_n) if user_id else []
        return {
            "content_indices":   content_idx,
            "content_distances": content_dist,
            "collab_tracks":     collab_tracks,
            "alpha":             alpha
        }