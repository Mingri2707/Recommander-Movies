import pandas as pd

# 1. Đọc dữ liệu từ các tệp .dat
# Lưu ý: encoding='latin-1' thường được dùng cho dataset MovieLens để tránh lỗi ký tự
movies = pd.read_csv(r'archive\movies.dat', sep='::', engine='python', 
                     names=['movie_id', 'title', 'genres'], encoding='latin-1')

ratings = pd.read_csv(r'archive\ratings.dat', sep='::', engine='python', 
                      names=['user_id', 'movie_id', 'rating', 'timestamp'], encoding='latin-1')

users = pd.read_csv(r'archive\users.dat', sep='::', engine='python', 
                    names=['user_id', 'gender', 'age', 'occupation', 'zip'], encoding='latin-1')

# 2. Xử lý dữ liệu ngầm định (Implicit Feedback)
# Cách 1: Coi tất cả các tương tác (bất kể rating bao nhiêu) là 1
# Cách 2 (Khuyên dùng): Chỉ coi những phim rated >= 4 là 'thích' (1), còn lại bỏ qua hoặc coi là 0
ratings['interaction'] = (ratings['rating'] >= 4).astype(int)

# 3. Gộp các bảng (Merge)
# Gộp ratings với movies trước
df = pd.merge(ratings, movies, on='movie_id')

# Sau đó gộp với users
dataset = pd.merge(df, users, on='user_id')

# 4. Feature Engineering (Trích xuất thêm tính năng)
# Trích xuất năm sản xuất từ tiêu đề phim: "Toy Story (1995)" -> 1995
dataset['release_year'] = dataset['title'].str.extract(r'\((\d{4})\)', expand=False)

# Xử lý genres: Chuyển thành list để dễ xử lý sau này
dataset['genres'] = dataset['genres'].str.split('|')

# 5. Sắp xếp theo thời gian để chia train/test (tránh rò rỉ dữ liệu tương lai)
dataset = dataset.sort_values('timestamp')

# Hiển thị kết quả gộp
print("Cấu trúc Master Dataset:")
print(dataset.head())

# 6. Lưu ra file CSV để dùng cho việc train
dataset.to_csv('master_dataset_for_train.csv', index=False)
print(dataset[dataset['movie_id'] == 1][['movie_id', 'title']].head(1))