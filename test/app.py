from flask import Flask, jsonify, send_from_directory
import pandas as pd
import json
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

app = Flask(__name__)

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
    
    # Start the recursive extraction
    extract_verses(data)
    return pd.DataFrame(verses)

def process_verses(df):
    analyzer = SentimentIntensityAnalyzer()
    
    # Get sentiment scores
    sentiments = df['text'].apply(lambda x: analyzer.polarity_scores(x))
    df['compound'] = sentiments.apply(lambda x: x['compound'])
    df['subjectivity'] = sentiments.apply(lambda x: (x['pos'] + x['neg']) / (x['pos'] + x['neg'] + x['neu'] + 0.0001))
    
    # Classify emotions
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

# Load and process data
df = load_quran_data()
df = process_verses(df)

@app.route('/')
def home():
    return send_from_directory('static', 'index.html')

@app.route('/api/verse/<emotion>')
def get_verse(emotion):
    emotion_verses = df[df['emotion'] == emotion]
    
    if emotion_verses.empty:
        return jsonify({'error': 'No verses found for this emotion'}), 404
    
    verse = emotion_verses.sample(1).iloc[0]
    
    return jsonify({
        'text': verse['text'],
        'surah_name': verse['surah_name'],
        'ayah_no': int(verse['ayah_no']),
        'confidence': float(verse['confidence'])
    })

if __name__ == '__main__':
    app.run(debug=True) 