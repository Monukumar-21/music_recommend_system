import streamlit as st
import requests
import pandas as pd

API_URL = "http://localhost:8000/recommend"

st.set_page_config(page_title="Music Recommender", page_icon="🎵", layout="centered")
st.title("Music Recommendation System")
st.markdown("Adjust audio features below to find similar tracks.")

with st.sidebar:
    st.header("Audio features")
    danceability     = st.slider("Danceability",     0.0, 1.0, 0.7)
    energy           = st.slider("Energy",           0.0, 1.0, 0.8)
    loudness         = st.slider("Loudness (dB)",   -60.0, 0.0, -5.0)
    speechiness      = st.slider("Speechiness",      0.0, 1.0, 0.05)
    acousticness     = st.slider("Acousticness",     0.0, 1.0, 0.1)
    instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0)
    liveness         = st.slider("Liveness",         0.0, 1.0, 0.1)
    valence          = st.slider("Valence",          0.0, 1.0, 0.6)
    tempo            = st.slider("Tempo (BPM)",      50.0, 200.0, 120.0)
    duration_ms      = st.slider("Duration (ms)",    60000, 600000, 210000)
    user_id          = st.text_input("User ID (optional)", "")
    top_n            = st.number_input("Recommendations", 1, 20, 10)
    alpha            = st.slider("Content vs Collab weight", 0.0, 1.0, 0.6)

if st.button("Get Recommendations"):
    payload = {
        "danceability": danceability,
        "energy": energy,
        "loudness": loudness,
        "speechiness": speechiness,
        "acousticness": acousticness,
        "instrumentalness": instrumentalness,
        "liveness": liveness,
        "valence": valence,
        "tempo": tempo,
        "duration_ms": duration_ms,
        "user_id": user_id if user_id else None,
        "top_n": top_n,
        "alpha": alpha
    }

    with st.spinner("Fetching recommendations..."):
        try:
            res = requests.post(API_URL, json=payload, timeout=10)
            res.raise_for_status()
            data = res.json()

            st.subheader("Content-based matches")
            df_content = pd.DataFrame({
                "Track index": data["content_indices"],
                "Distance (lower = closer)": [round(d, 4) for d in data["content_distances"]]
            })
            st.dataframe(df_content, use_container_width=True)

            if data["collab_tracks"]:
                st.subheader("Collaborative filter picks")
                st.write(data["collab_tracks"])

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to the API. Make sure `uvicorn api.main:app` is running.")
        except Exception as e:
            st.error(f"Error: {e}")