import requests

# Địa chỉ server Flask của bạn
BASE_URL = "http://127.0.0.1:5000/recommend"

def test_recommendation(user_id=None):
    """
    Hàm test gửi request đến Flask API và in ra ID + Tên phim gợi ý
    """
    params = {"user_id": user_id} if user_id else {}

@app.route('/recommend', methods=['GET'])
def recommend():
    raw_user_id = request.args.get('user_id')
    
    # TRƯỜNG HỢP: KHÁCH VÃNG LAI HOẶC YÊU CẦU NGẪU NHIÊN
    if not raw_user_id or int(raw_user_id) not in user2user_encoded:
        # Lấy toàn bộ danh sách movie_id từ movie_map
        all_ids = list(movie_map.keys())
        # Chọn ngẫu nhiên 5 phim
        random_movie_keys = random.sample(all_ids, min(len(all_ids), 5))
        
        recommendations = []
        for k in random_movie_keys:
            info = movie_map[k]
            recommendations.append({
                "movie_id": int(info['movie_id']),
                "title": info['title'],
                "score": "N/A (Random)"
            })
            
        return jsonify({
            "status": "guest_random",
            "message": "Đây là một vài bộ phim ngẫu nhiên dành cho bạn",
            "recommendations": recommendations
        })
    
    try:
        response = requests.get(BASE_URL, params=params)
        
        # Kiểm tra nếu server phản hồi lỗi
        if response.status_code != 200:
            print(f"\n[!] Lỗi ({response.status_code}): {response.json().get('error', 'Unknown error')}")
            return

        data = response.json()
        status = data.get("status", "unknown")
        
        print(f"\n" + "="*50)
        print(f" KIỂM TRA CHO USER_ID: {user_id if user_id else 'Khách vãng lai'}")
        print(f" Trạng thái: {status.upper()}")
        print("-"*50)
        print(f"{'STT':<5} | {'ID Phim':<10} | {'Tên Phim'}")
        print("-"*50)

        recommendations = data.get('recommendations', [])

        for i, rec in enumerate(recommendations, 1):
            if isinstance(rec, dict):
                # Trường hợp User thành viên (có đủ id, title, score)
                m_id = rec.get('movie_id', 'N/A')
                title = rec.get('title', 'Unknown')
                print(f"{i:<5} | {m_id:<10} | {title}")
            else:
                # Trường hợp Khách vãng lai (nếu chỉ trả về list string tên phim)
                print(f"{i:<5} | {'N/A':<10} | {rec}")
        
        print("="*50)

    except requests.exceptions.ConnectionError:
        print("\n[!] Lỗi: Không thể kết nối đến Flask server. Hãy chắc chắn bạn đã chạy python app.py")
    except Exception as e:
        print(f"\n[!] Đã xảy ra lỗi không xác định: {e}")

if __name__ == "__main__":
    # --- KỊCH BẢN TEST ---

    # 1. Test với user ID cụ thể (Thay bằng ID thực tế trong file CSV của bạn)
    # Ví dụ: 1, 10, 50...
    test_recommendation(user_id=1)
    
    # 2. Test với một ID chưa từng xuất hiện (Check logic khách vãng lai)
    test_recommendation(user_id=999999)
    
    # 3. Test trường hợp không truyền tham số (Gợi ý phim hot)
    test_recommendation(user_id=None)