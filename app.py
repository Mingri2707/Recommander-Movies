from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import pickle

app = Flask(__name__)

# --- Định nghĩa Model (Giữ nguyên cấu trúc) ---
class RecommenderNet(models.Model):
    def __init__(self, num_users, num_movies, embedding_size=50):
        super(RecommenderNet, self).__init__()
        self.user_embedding = layers.Embedding(num_users, embedding_size, embeddings_initializer="he_normal")
        self.movie_embedding = layers.Embedding(num_movies, embedding_size, embeddings_initializer="he_normal")
        self.concatenate = layers.Concatenate()
        self.dense1 = layers.Dense(64, activation="relu")
        self.dense2 = layers.Dense(32, activation="relu")
        self.output_layer = layers.Dense(1, activation="sigmoid")

    def call(self, inputs):
        u = self.user_embedding(inputs[:, 0])
        m = self.movie_embedding(inputs[:, 1])
        x = self.concatenate([u, m])
        x = self.dense1(x)
        x = self.dense2(x)
        return self.output_layer(x)

# --- Load dữ liệu ---
with open('model_artifacts.pkl', 'rb') as f:
    artifacts = pickle.load(f)

user2user_encoded = artifacts['user2user_encoded']
movie2movie_encoded = artifacts['movie2movie_encoded']
movie_map = artifacts['movie_map']
# Giả sử bạn đã lưu danh sách phim phổ biến vào artifacts
# Nếu chưa, bạn có thể tạo thủ công một list các ID phim hot nhất
popular_movies = artifacts.get('popular_movies', []) 

model = RecommenderNet(artifacts['num_users'], artifacts['num_movies'])
model.build(input_shape=(None, 2))
model.load_weights('recommender_weights.weights.h5')

@app.route('/recommend', methods=['GET'])
def recommend():
    raw_user_id = request.args.get('user_id')
    
    # TRƯỜNG HỢP: KHÁCH VÃNG LAI (Không có ID hoặc ID lạ)
    if not raw_user_id or int(raw_user_id) not in user2user_encoded:
        return jsonify({
            "status": "guest",
            "message": "Gợi ý những phim phổ biến nhất cho bạn",
            "recommendations": popular_movies[:10] # Trả về 10 phim hot
        })

    # TRƯỜNG HỢP: USER ĐÃ CÓ TRONG HỆ THỐNG
    user_encoded = user2user_encoded[int(raw_user_id)]
    all_movie_ids = np.array(list(movie2movie_encoded.values()))
    user_input = np.array([user_encoded] * len(all_movie_ids))
    predict_input = np.stack([user_input, all_movie_ids], axis=1)

    ratings = model.predict(predict_input, verbose=0).flatten()
    top_indices = ratings.argsort()[-10:][::-1]
    
    recommendations = []
    for idx in top_indices:
        m_encoded = all_movie_ids[idx]
        info = movie_map[m_encoded]
        recommendations.append({
            "movie_id": int(info['movie_id']),
            "title": info['title'],
            "score": round(float(ratings[idx]), 4)
        })

    return jsonify({
        "status": "member",
        "user_id": raw_user_id,
        "recommendations": recommendations
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)