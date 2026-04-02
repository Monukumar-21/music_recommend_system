from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import sys
sys.path.append("../src")
from recommend import MusicRecommender

app = FastAPI(title="Music Recommender API", version="1.0")
recommender = MusicRecommender()

class TrackFeatures(BaseModel):
    danceability: float
    energy: float
    loudness: float
    speechiness: float
    acousticness: float
    instrumentalness: float
    liveness: float
    valence: float
    tempo: float
    duration_ms: float
    user_id: Optional[str] = None
    top_n: int = 10
    alpha: float = 0.6

class RecommendationResponse(BaseModel):
    content_indices: List[int]
    content_distances: List[float]
    collab_tracks: List[str]
    alpha: float

@app.get("/")
def root():
    return {"message": "Music Recommender API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/recommend", response_model=RecommendationResponse)
def recommend(track: TrackFeatures):
    try:
        features = [
            track.danceability, track.energy, track.loudness,
            track.speechiness, track.acousticness, track.instrumentalness,
            track.liveness, track.valence, track.tempo, track.duration_ms
        ]
        result = recommender.hybrid_recommend(
            features=features,
            user_id=track.user_id,
            top_n=track.top_n,
            alpha=track.alpha
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))