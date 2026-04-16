import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pickle

# --- 1. Load và Tiền xử lý dữ liệu ---
df = pd.read_csv('master_dataset_for_train.csv')

# Encoding IDs
user_ids = df['user_id'].unique().tolist()
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
movie_ids = df['movie_id'].unique().tolist()
movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}

df['user'] = df['user_id'].map(user2user_encoded)
df['movie'] = df['movie_id'].map(movie2movie_encoded)

num_users = len(user2user_encoded)
num_movies = len(movie2movie_encoded)

# --- 2. Kỹ thuật Negative Sampling (Tăng mạnh NDCG) ---
def create_negative_samples(df, num_negatives=4):
    user_movie_set = df.groupby('user')['movie'].apply(set).to_dict()
    all_movies = np.array(list(movie2movie_encoded.values()))
    
    neg_users, neg_movies, neg_labels = [], [], []

    print(f"Đang tạo Negative Samples (Tỉ lệ 1:{num_negatives})...")
    
    unique_users = df['user'].unique()
    
    for u in unique_users:
        pos_movies = user_movie_set[u]
        # Tìm danh sách các phim mà User CHƯA xem
        # Dùng set để trừ đi cho nhanh
        possible_negatives = np.setdiff1d(all_movies, list(pos_movies))
        
        # Số lượng mẫu âm cần lấy cho user này
        n_neg = len(pos_movies) * num_negatives
        
        # Nếu số lượng phim chưa xem ít hơn yêu cầu, lấy tất cả phim chưa xem
        n_to_sample = min(len(possible_negatives), n_neg)
        
        if n_to_sample > 0:
            sampled_negs = np.random.choice(possible_negatives, size=n_to_sample, replace=False)
            
            neg_users.extend([u] * n_to_sample)
            neg_movies.extend(sampled_negs)
            neg_labels.extend([0] * n_to_sample)
            
    neg_df = pd.DataFrame({'user': neg_users, 'movie': neg_movies, 'interaction': neg_labels})
    print(f"Đã tạo xong {len(neg_df)} mẫu âm.")
    return pd.concat([df[['user', 'movie', 'interaction']], neg_df], ignore_index=True)

# Tạo data mới với tỉ lệ 1 Positive : 4 Negative
df_expanded = create_negative_samples(df)

X = df_expanded[['user', 'movie']].values
y = df_expanded['interaction'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# --- 3. Xây dựng Model Cải tiến (Deep NCF) ---
class RecommenderNet(models.Model):
    def __init__(self, num_users, num_movies, embedding_size=64):
        super(RecommenderNet, self).__init__()
        
        # User & Movie Embeddings
        self.user_embedding = layers.Embedding(
            num_users, embedding_size, 
            embeddings_initializer="he_normal",
            embeddings_regularizer=tf.keras.regularizers.l2(1e-6)
        )
        self.movie_embedding = layers.Embedding(
            num_movies, embedding_size, 
            embeddings_initializer="he_normal",
            embeddings_regularizer=tf.keras.regularizers.l2(1e-6)
        )
        
        # Mạng Neural sâu hơn
        self.concatenate = layers.Concatenate()
        self.bn0 = layers.BatchNormalization()
        
        self.dense1 = layers.Dense(128, activation="relu")
        self.bn1 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(0.3)
        
        self.dense2 = layers.Dense(64, activation="relu")
        self.bn2 = layers.BatchNormalization()
        self.dropout2 = layers.Dropout(0.2)
        
        self.dense3 = layers.Dense(32, activation="relu")
        self.output_layer = layers.Dense(1, activation="sigmoid")

    def call(self, inputs):
        u = self.user_embedding(inputs[:, 0])
        m = self.movie_embedding(inputs[:, 1])
        
        x = self.concatenate([u, m])
        x = self.bn0(x)
        
        x = self.dense1(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        
        x = self.dense2(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        
        x = self.dense3(x)
        return self.output_layer(x)

model = RecommenderNet(num_users, num_movies)
# Dùng Learning Rate thấp hơn để học kĩ hơn
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# Callbacks thông minh
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2) # Tự giảm LR nếu đứng máy
]

# --- 4. Huấn luyện ---
history = model.fit(
    X_train, y_train,
    batch_size=512,
    epochs=30,
    validation_data=(X_test, y_test),
    callbacks=callbacks
)

# --- 5. Lưu Artifacts ---
model.save_weights('recommender_weights.weights.h5')
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

print("Hoàn tất huấn luyện model cải tiến!")

import math

def evaluate_model(model, df_test, movie2movie_encoded, k_list=[10, 20, 50]):
    """
    Đánh giá Hit Rate và NDCG cho nhiều giá trị K cùng lúc.
    """
    # Chỉ đánh giá trên các tương tác thực tế (nhãn 1) trong tập test
    test_data_pos = df_test[df_test['interaction'] == 1]
    test_users = test_data_pos['user'].unique()
    
    # Giới hạn số user đánh giá nếu tập test quá lớn (để tiết kiệm thời gian)
    if len(test_users) > 500:
        test_users = np.random.choice(test_users, 500, replace=False)

    all_movie_ids = np.array(list(movie2movie_encoded.values()))
    
    # Khởi tạo từ điển lưu kết quả
    results = {f"HR@{k}": [] for k in k_list}
    results.update({f"NDCG@{k}": [] for k in k_list})

    print(f"--- Đang đánh giá trên {len(test_users)} users ---")

    for u in test_users:
        # 1. Lấy danh sách phim thực tế user này thích trong tập test
        actual_liked = set(test_data_pos[test_data_pos['user'] == u]['movie'].values)
        
        # 2. Dự đoán điểm cho TẤT CẢ các phim
        u_input = np.full(len(all_movie_ids), u)
        predict_input = np.stack([u_input, all_movie_ids], axis=1)
        
        predictions = model.predict(predict_input, batch_size=1024, verbose=0).flatten()
        
        # 3. Lấy chỉ số của các phim có điểm cao nhất (xếp hạng)
        # Sắp xếp giảm dần
        top_indices = predictions.argsort()[::-1] 
        
        # 4. Tính toán cho từng K
        for k in k_list:
            top_k_movies = top_indices[:k]
            
            # --- Tính Hit Rate ---
            hit = any(m in actual_liked for m in top_k_movies)
            results[f"HR@{k}"].append(1.0 if hit else 0.0)
            
            # --- Tính NDCG ---
            dcg = 0.0
            idcg = 0.0
            
            # Tính DCG
            for i, m in enumerate(top_k_movies):
                if m in actual_liked:
                    dcg += 1.0 / math.log2(i + 2)
            
            # Tính IDCG (Trường hợp lý tưởng: các phim thích nằm hết ở đầu)
            num_relevant = len(actual_liked)
            for i in range(min(num_relevant, k)):
                idcg += 1.0 / math.log2(i + 2)
            
            ndcg = (dcg / idcg) if idcg > 0 else 0.0
            results[f"NDCG@{k}"].append(ndcg)

    # In kết quả tổng hợp
    print("\n" + "="*40)
    print(f"{'Metric':<15} | {'Score':<10}")
    print("-" * 40)
    for metric, values in results.items():
        print(f"{metric:<15} | {np.mean(values)*100:.2f}%")
    print("="*40)

# --- Gọi hàm sau khi đã huấn luyện xong ---
# k_list là danh sách các ngưỡng bạn muốn xem (10, 20, 30, 50...)
df_test = pd.DataFrame(X_test, columns=['user', 'movie'])
df_test['interaction'] = y_test

# Bây giờ gọi hàm đánh giá sẽ không còn lỗi nữa
evaluate_model(model, df_test, movie2movie_encoded, k_list=[10, 20, 30, 50])