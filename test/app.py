from flask import Flask, jsonify, send_from_directory, request
import pandas as pd
import json
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from textblob import TextBlob
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

app = Flask(__name__)

EMOTION_MAPPING = {
    'angry': 'peaceful',
    'fearful': 'joyful',
    'remorseful': 'reflective',
    'reflective': 'joyful',
    'joyful': 'remorseful',
    'peaceful': 'remorseful'
}

def load_quran_data():
    with open('quran.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    verses = []
    
    def extract_verses(item):
        if isinstance(item, dict):
            if 'children' in item:
                for child in item['children']:
                    extract_verses(child)
            elif 'ayah_en' in item:
                verses.append({
                    'text': item['ayah_en'],
                    'surah_name': item['surah_name_en'],
                    'ayah_no': item['ayah_no_surah']
                })
    
    extract_verses(data)
    return pd.DataFrame(verses)

def process_verses(df):
    analyzer = SentimentIntensityAnalyzer()
    
    sentiments = df['text'].apply(lambda x: analyzer.polarity_scores(x))
    df['compound'] = sentiments.apply(lambda x: x['compound'])
    df['subjectivity'] = sentiments.apply(lambda x: (x['pos'] + x['neg']) / (x['pos'] + x['neg'] + x['neu'] + 0.0001))
    
    def get_emotion_category(row):
        compound = row['compound']
        subjectivity = row['subjectivity']
        
        if compound > 0.2:
            return 'joyful' if subjectivity > 0.5 else 'peaceful'
        elif compound < -0.2:
            return 'angry' if subjectivity > 0.5 else 'fearful'
        else:
            return 'remorseful' if subjectivity > 0.5 else 'reflective'
    
    df['emotion'] = df.apply(get_emotion_category, axis=1)
    df['confidence'] = df['compound'].abs()
    
    return df

df = load_quran_data()
df = process_verses(df)

@app.route('/')
def home():
    return send_from_directory('static', 'index.html')

@app.route('/api/verse/<emotion>')
def get_verse(emotion):
    counter_emotion = EMOTION_MAPPING.get(emotion, emotion)
    emotion_verses = df[df['emotion'] == counter_emotion]
    
    if emotion_verses.empty:
        return jsonify({'error': f'No verses found for emotion {counter_emotion}'}), 404
    
    verse = emotion_verses.sample(1).iloc[0]
    
    return jsonify({
        'original_emotion': emotion,
        'counterbalancing_emotion': counter_emotion,
        'text': verse['text'],
        'surah_name': verse['surah_name'],
        'ayah_no': int(verse['ayah_no']),
        'confidence': float(verse['confidence'])
    })

@app.route('/api/analyze-emotion', methods=['POST'])
def analyze_emotion():
    text = request.json.get('text', '')
    
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    if polarity > 0.3:
        emotion = 'joyful' if subjectivity > 0.5 else 'peaceful'
    elif polarity < -0.3:
        emotion = 'angry' if subjectivity > 0.5 else 'fearful'
    else:
        emotion = 'remorseful' if subjectivity > 0.5 else 'reflective'
    
    counter_emotion = EMOTION_MAPPING.get(emotion, 'reflective')
    emotion_verses = df[df['emotion'] == counter_emotion]
    
    if emotion_verses.empty:
        recommended_verse = None
    else:
        verse = emotion_verses.sample(1).iloc[0]
        recommended_verse = {
            'text': verse['text'],
            'surah_name': verse['surah_name'],
            'ayah_no': int(verse['ayah_no']),
            'confidence': float(verse['confidence'])
        }
    
    return jsonify({
        'emotion': emotion,
        'confidence': abs(polarity),
        'counterbalancing_emotion': counter_emotion,
        'recommended_verse': recommended_verse
    })

if __name__ == '__main__':
    app.run(debug=True) 