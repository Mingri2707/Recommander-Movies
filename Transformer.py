import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

# ==========================================
# 1. TIỀN XỬ LÝ DỮ LIỆU
# ==========================================
# Giả định file 'master_dataset_for_train.csv' đã có từ bước merge trước đó
df = pd.read_csv('master_dataset_for_train.csv')

# Encode ID về dạng liên tục (bắt buộc cho Embedding Layer)
user_map = {x: i for i, x in enumerate(df["user_id"].unique())}
movie_map = {x: i for i, x in enumerate(df["movie_id"].unique())}
inverse_movie_map = {i: x for x, i in movie_map.items()} # Để map ngược lại ID gốc

df["user_idx"] = df["user_id"].map(user_map)
df["movie_idx"] = df["movie_id"].map(movie_map)

# Sắp xếp theo thời gian để tạo chuỗi hành vi
df = df.sort_values(by=["user_idx", "timestamp"])

SEQ_LEN = 10 # Độ dài chuỗi (lấy 10 phim trước để đoán phim tiếp theo)

def create_sequences(df, window_size):
    user_groups = df.groupby("user_idx")["movie_idx"].apply(list)
    X, y = [], []
    for user_id, items in user_groups.items():
        if len(items) > window_size:
            # Tạo các cửa sổ trượt (Sliding Window)
            for i in range(len(items) - window_size):
                X.append(items[i : i + window_size])
                y.append(items[i + window_size])
    return np.array(X), np.array(y)

X, y = create_sequences(df, SEQ_LEN)

# Chia tập dữ liệu: 80% Train, 10% Val, 10% Test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# ==========================================
# 2. XÂY DỰNG MÔ HÌNH TRANSFORMER (BST)
# ==========================================
num_movies = len(movie_map)
EMBED_DIM = 64
NUM_HEADS = 4

def build_bst_model():
    inputs = layers.Input(shape=(SEQ_LEN,), name="movie_sequence")
    
    # 1. Movie Embedding
    movie_emb = layers.Embedding(input_dim=num_movies, output_dim=EMBED_DIM)(inputs)
    
    # 2. Positional Encoding (Cực kỳ quan trọng cho Transformer)
    positions = tf.range(start=0, limit=SEQ_LEN, delta=1)
    pos_emb = layers.Embedding(input_dim=SEQ_LEN, output_dim=EMBED_DIM)(positions)
    
    # Kết hợp thông tin phim và thứ tự thời gian
    x = movie_emb + pos_emb 

    # 3. Transformer Block
    # Self-attention giúp học mối quan hệ giữa các phim trong chuỗi
    attention_output = layers.MultiHeadAttention(
        num_heads=NUM_HEADS, key_dim=EMBED_DIM
    )(x, x)
    x = layers.LayerNormalization()(x + attention_output)
    
    # Feed Forward Network
    ffn = layers.Dense(EMBED_DIM, activation="relu")(x)
    x = layers.LayerNormalization()(x + ffn)
    
    # 4. Output Layer
    x = layers.Flatten()(x)
    # Dự đoán xác suất cho TẤT CẢ các phim có trong hệ thống
    outputs = layers.Dense(num_movies, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

model = build_bst_model()
model.compile(
    optimizer="adam", 
    loss="sparse_categorical_crossentropy", 
    metrics=["accuracy"]
)

early_stopping = EarlyStopping(
    monitor='val_loss',       
    patience=5,              
    restore_best_weights=True 
)

# Huấn luyện mô hình
print("Bắt đầu huấn luyện...")
model.fit(
    X_train, y_train, 
    validation_data=(X_val, y_val), 
    epochs=20, 
    batch_size=256,
    callbacks=[early_stopping]
)

# ==========================================
# 3. ĐÁNH GIÁ VỚI HIT RATE@K
# ==========================================
def evaluate_hit_rate(model, X_test, y_test, k):
    print(f"Đang tính toán Hit Rate@{k}...")
    # Dự đoán xác suất cho toàn bộ tập test
    preds = model.predict(X_test, batch_size=512)
    
    # Lấy top k phim có xác suất cao nhất
    top_k_indices = tf.math.top_k(preds, k=k).indices.numpy()
    
    hits = 0
    for i in range(len(y_test)):
        if y_test[i] in top_k_indices[i]:
            hits += 1
            
    return hits / len(y_test)

hr_10 = evaluate_hit_rate(model, X_test, y_test, k=10)
hr_50 = evaluate_hit_rate(model, X_test, y_test, k=50)

print("-" * 30)
print(f"KẾT QUẢ ĐÁNH GIÁ:")
print(f"Hit Rate @ 10: {hr_10:.4f}")
print(f"Hit Rate @ 50 : {hr_50:.4f}")
print("-" * 30)

# Ví dụ gợi ý thực tế:
# Lấy 1 chuỗi phim từ tập test, dự đoán ID gốc của phim tiếp theo
sample_seq = X_test[0:1]
prediction = model.predict(sample_seq)
top_1_idx = np.argmax(prediction)
original_movie_id = inverse_movie_map[top_1_idx]
print(f"Dự đoán phim tiếp theo (ID gốc): {original_movie_id}")