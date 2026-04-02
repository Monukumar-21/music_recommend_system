# Spotify Recommendation System

A hybrid music recommendation system combining collaborative filtering, content-based filtering, and deep learning approaches.

## 🎯 Features

- **Collaborative Filtering**: Matrix factorization using SVD to find similar users
- **Content-Based Filtering**: KNN-based recommendations using track audio features
- **Deep Learning**: Autoencoder for feature learning and dimensionality reduction
- **Hybrid Approach**: Combines multiple methods for improved recommendations
- **REST API**: FastAPI backend for easy integration
- **Web Interface**: Streamlit frontend for interactive recommendations

## 📁 Project Structure

```
spotify_recommendation system/
├── data/
│   └── tracks.csv              # Dataset with Spotify tracks and features
├── models/
│   ├── autoencoder.pt          # Trained autoencoder weights
│   ├── knn_model.pkl           # Content-based KNN model
│   ├── svd_model.pkl           # Collaborative filtering (SVD) model
│   └── scaler.pkl              # Fitted StandardScaler
├── src/
│   ├── preprocess.py           # Feature engineering and scaling
│   ├── train_collab.py         # Collaborative filtering training
│   ├── train_content.py        # Content-based + autoencoder training
│   └── recommend.py            # Unified hybrid recommender class
├── api/
│   └── main.py                 # FastAPI application
├── app/
│   └── streamlit_app.py        # Streamlit web interface
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
cd spotify_recommendation system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

Place your Spotify dataset in `data/tracks.csv` with the following structure:
- `track_id`: Unique track identifier
- `track_name`: Track name
- `artist`: Artist name
- `features`: Audio features (danceability, energy, valence, etc.)

### 3. Train Models

```bash
# Preprocess data
python src/preprocess.py

# Train collaborative filtering
python src/train_collab.py

# Train content-based and autoencoder
python src/train_content.py
```

### 4. Run Backend API

```bash
cd api
python -m uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

### 5. Run Frontend

```bash
streamlit run app/streamlit_app.py
```

The web interface will open at `http://localhost:8501`

## 📚 API Endpoints

### Health Check
- `GET /health` - API health status

### Get Recommendations
- `POST /recommendations`
  ```json
  {
    "user_id": 1,
    "track_id": 123,
    "n_recommendations": 10
  }
  ```

## 🔧 Configuration

Adjust model parameters in the training scripts:

- **SVD**: Number of latent factors (`n_components`)
- **KNN**: Number of neighbors (`n_neighbors`)
- **Autoencoder**: Layer sizes and encoding dimension
- **Hybrid**: Weights for each recommendation method

## 📊 Data Format

Expected CSV columns:
- `track_id`: Integer
- `track_name`: String
- `artist`: String
- `danceability`: Float (0-1)
- `energy`: Float (0-1)
- `valence`: Float (0-1)
- `... other audio features`

## 🎓 Recommendation Methods

### Collaborative Filtering (SVD)
Matrix factorization to find patterns in user-item interactions.

### Content-Based (KNN)
Recommends tracks similar to a given track based on audio features.

### Deep Learning (Autoencoder)
Learns compressed representations of tracks for similarity matching.

### Hybrid
Combines predictions from all three methods with configurable weights.

## 📦 Dependencies

- **pandas**: Data manipulation
- **scikit-learn**: ML algorithms and preprocessing
- **torch**: Deep learning framework
- **fastapi**: API framework
- **streamlit**: Web interface
- **numpy/scipy**: Numerical computing

## 🤝 Contributing

Contributions are welcome! Please fork and submit pull requests.

## 📝 License

MIT License

## 📞 Support

For issues or questions, please open an issue on the repository.

---

**Built with ❤️ for music lovers and data scientists**
