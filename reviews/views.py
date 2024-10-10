import os
import joblib
import re
import string

from django.shortcuts import render
from django.conf import settings
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from .forms import ReviewForm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'vectorizer.pkl')


try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
except Exception as e:
    print(f"Ошибка при загрузке модели или векторизатора: {e}")
    model = None
    vectorizer = None


stop_words = set(ENGLISH_STOP_WORDS)
stop_words.update(['br', 'movie', 'film'])

def preprocess_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def review_view(request):
    if request.method == 'POST':
        form = ReviewForm(request.POST)
        if form.is_valid():
            review_text = form.cleaned_data['review_text']
            processed_text = preprocess_text(review_text)
            if model and vectorizer:
                try:
                    vectorized_text = vectorizer.transform([processed_text])
                    sentiment = model.predict(vectorized_text)[0]
                    proba = model.predict_proba(vectorized_text)[0]
                    positive_proba = proba[1]
                    rating = int(positive_proba * 9 + 1)  
                    status = 'Положительный' if sentiment == 1 else 'Отрицательный'
                    return render(request, 'reviews/result.html', {'rating': rating, 'status': status})
                except Exception as e:
                    
                    return render(request, 'reviews/error.html', {'message': f'Ошибка при анализе отзыва: {e}'})
            else:
                
                return render(request, 'reviews/error.html', {'message': 'Модель не загружена'})
    else:
        form = ReviewForm()
    return render(request, 'reviews/review_form.html', {'form': form})
