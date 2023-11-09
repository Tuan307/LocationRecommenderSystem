# views.py

from django.http import JsonResponse
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
def get_user_data(userId):
    # Gọi API lấy thông tin user
    user_api_url = 'http://127.0.0.1:8080/api/v1/users/'+userId
    response = requests.get(user_api_url).json()

    return response['data'] if 'data' in response else None

def preprocess_text(text):
    # Chuyển chuỗi thành chữ thường
    text = text.lower()
    
    # Thực hiện các bước tiền xử lý văn bản khác ở đây, ví dụ: loại bỏ dấu câu, ...
    
    return text    
def get_travel_data():
    # Gọi API lấy thông tin địa điểm du lịch
    travel_api_url = 'http://localhost:8080/api/v1/city/all'
    response = requests.get(travel_api_url).json()

    return response['data'] if 'data' in response else None

def recommend_cities(user_data, city_data):
    # Tiền xử lý dữ liệu
    city_descriptions = [preprocess_text(city['description']) for city in city_data]
    city_tags = [preprocess_text(city['tag']) for city in city_data]
    user_interests = " ".join(interest['tourInterest']['name'].lower() for interest in user_data['userInterestProfiles'])

    # Tạo DataFrame cho dữ liệu thành phố
    data = {
        'city_id': [city['cityId'] for city in city_data],
        'description': city_descriptions,
        'tag': city_tags
    }
    df = pd.DataFrame(data)

    # Tạo vectorizer cho TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['description'] + " " + df['tag'])

    # Tính độ tương tự cosine giữa sở thích của người dùng và các thành phố
    user_vector = vectorizer.transform([user_interests])
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix)
    
    # Lấy danh sách các thành phố và điểm số tương tự
    city_indices = list(range(len(city_data)))
    similarity_scores1 = similarity_scores[0]

    # Sắp xếp các thành phố dựa trên độ tương tự từ cao xuống thấp
    city_scores = [(city_data[i], similarity_scores1[i]) for i in city_indices]
    city_scores.sort(key=lambda x: x[1], reverse=True)

    # Gợi ý danh sách thành phố
    recommended_cities = []
    for city, score in city_scores:
        recommended_cities.append({
            'city_id': city['cityId'],
            'city_name': city['cityName'],
            'image_url': city['cityImages'],
            'description': city['description'],
            'tag': city['tag'],
            'similarity_score': str(score)
        })

    return recommended_cities

def recommendation(request,userId):
    user_data = get_user_data(userId)
    city_data = get_travel_data()

    if user_data and city_data:
        recommended_cities = recommend_cities(user_data, city_data)
        return JsonResponse({'recommended_cities': recommended_cities})

    return JsonResponse({'error': 'Failed to fetch data from APIs.'})