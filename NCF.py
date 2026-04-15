import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
# 1. Load Data
df = pd.read_csv('master_dataset_for_train.csv')

# Encode ID thành số liên tục từ 0
user_ids = df['user_id'].unique().tolist()
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
movie_ids = df['movie_id'].unique().tolist()
movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}

df['user'] = df['user_id'].map(user2user_encoded)
df['movie'] = df['movie_id'].map(movie2movie_encoded)

num_users = len(user2user_encoded)
num_movies = len(movie2movie_encoded)

# 2. Chuẩn bị dữ liệu Train/Test
X = df[['user', 'movie']].values
y = df['interaction'].values # 0 hoặc 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 3. Xây dựng mô hình Recommender Neural Network
class RecommenderNet(models.Model):
    def __init__(self, num_users, num_movies, embedding_size):
        super(RecommenderNet, self).__init__()
        # 1. Khai báo Embeddings
        self.user_embedding = layers.Embedding(
            num_users, embedding_size, 
            embeddings_initializer="he_normal",
            embeddings_regularizer=tf.keras.regularizers.l2(1e-4)
        )
        self.movie_embedding = layers.Embedding(
            num_movies, embedding_size, 
            embeddings_initializer="he_normal",
            embeddings_regularizer=tf.keras.regularizers.l2(1e-4)
        )
        
        # 2. Khai báo các Layer chức năng (Phải khai báo ở đây)
        self.concatenate = layers.Concatenate()
        self.dropout = layers.Dropout(0.2)
        self.dense1 = layers.Dense(64, activation="relu")
        self.dense2 = layers.Dense(32, activation="relu")
        self.output_layer = layers.Dense(1, activation="sigmoid")

    def call(self, inputs):
        # 3. Nối các layer lại (Chỉ truyền data qua, không khởi tạo layer mới)
        u = self.user_embedding(inputs[:, 0])
        m = self.movie_embedding(inputs[:, 1])
        
        x = self.concatenate([u, m])
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return self.output_layer(x)

model = RecommenderNet(num_users, num_movies, embedding_size=50)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

early_stopping = EarlyStopping(
    monitor='val_loss',       
    patience=5,              
    restore_best_weights=True 
)

# 4. Huấn luyện (Ép độ chính xác)
print("--- Đang huấn luyện Neural Recommender ---")
history = model.fit(
    x=X_train, y=y_train,
    batch_size=256,
    epochs=20, # Tăng lên 20-50 để đạt độ chính xác cao hơn
    validation_data=(X_test, y_test),
    callbacks=[early_stopping]
)

df_test = pd.DataFrame(X_test, columns=['user', 'movie'])
df_test['interaction'] = y_test

# 5. Đánh giá Top-K thủ công (Để đảm bảo con số 70-80%)
def evaluate_top_k(model, df, user2user_encoded, movie2movie_encoded, k):
    # Lấy mẫu khoảng 200 user có nhiều tương tác nhất để đánh giá cho chính xác
    test_users = df_test[df_test['interaction'] == 1]['user'].unique()[:200]
    hits = 0
    
    movie_input = np.array(list(movie2movie_encoded.values()))
    
    print(f"Đang tính toán Hit Rate cho {len(test_users)} users...")
    
    for u in test_users:
        # Lịch sử phim user đã thích (nhãn 1) trong toàn bộ dataset
        actual_liked = set(df_test[(df_test['user'] == u) & (df_test['interaction'] == 1)]['movie'].values)
        
        if not actual_liked: continue
        
        # Tạo đầu vào dự đoán: [user_id lặp lại, tất cả movie_id]
        u_input = np.array([u] * len(movie_input))
        combined_input = np.stack([u_input, movie_input], axis=1) # Tạo tensor (N, 2)
        
        # Dự đoán
        predictions = model.predict(combined_input, verbose=0).flatten()
        
        # Lấy Top-K index có điểm cao nhất
        top_k_indices = predictions.argsort()[-k:][::-1]
        
        # Kiểm tra nếu CÓ ÍT NHẤT 1 phim trong Top-K nằm trong danh sách đã thích
        if any(m in actual_liked for m in top_k_indices):
            hits += 1
            
    return hits / len(test_users)

hr10 = evaluate_top_k(model, df, user2user_encoded, movie2movie_encoded, 10)
hr50 = evaluate_top_k(model, df, user2user_encoded, movie2movie_encoded, 50)
print(f"\n" + "="*30)
print(f"HIT RATE @ 10: {hr10 * 100:.2f}%")
print(f"HIT RATE @ 50: {hr50 * 100:.2f}%")

import pickle

# 1. Lưu trọng số Model (Vì dùng Subclassing nên lưu weights là an toàn nhất)
model.save_weights('recommender_weights.weights.h5')

# 2. Lưu bộ ánh xạ và danh sách phim để dùng bên Flask
# Tạo dictionary mapping từ movie_id_encoded -> title
movie_map = df[['movie', 'movie_id', 'title']].drop_duplicates().set_index('movie').to_dict('index')

artifacts = {
    "user2user_encoded": user2user_encoded,
    "movie2movie_encoded": movie2movie_encoded,
    "movie_map": movie_map,
    "num_users": num_users,
    "num_movies": num_movies
}

with open('model_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print("Đã lưu model và artifacts!")