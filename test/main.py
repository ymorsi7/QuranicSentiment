import json
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Load Quran data
def load_quran_data(file_path='quran.json'):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract verses
    verses = []
    def extract_verses(item):
        if 'children' in item:
            for child in item['children']:
                extract_verses(child)
        elif 'ayah_en' in item:
            verses.append({
                'surah_name': item['surah_name_en'],
                'ayah_no': item['ayah_no_surah'],
                'text': item['ayah_en']
            })
    
    extract_verses(data)
    return pd.DataFrame(verses)

# Enhanced sentiment analysis
def analyze_sentiment_enhanced(text):
    # VADER sentiment
    vader = SentimentIntensityAnalyzer()
    vader_scores = vader.polarity_scores(text)
    
    # TextBlob sentiment
    blob = TextBlob(text)
    
    # Word-level analysis
    words = word_tokenize(text)
    pos_tags = nltk.pos_tag(words)
    
    # Count important POS tags that might indicate emotional content
    imperatives = sum(1 for _, tag in pos_tags if tag.startswith('VB'))
    adjectives = sum(1 for _, tag in pos_tags if tag.startswith('JJ'))
    
    # Calculate confidence score based on multiple factors
    confidence = (abs(vader_scores['compound']) * 0.4 +  # Strong sentiment indicates higher confidence
                 blob.sentiment.subjectivity * 0.3 +    # Higher subjectivity often indicates clearer emotion
                 (imperatives / len(words) if words else 0) * 0.15 +  # Presence of imperatives
                 (adjectives / len(words) if words else 0) * 0.15)    # Presence of descriptive words
    
    return {
        'compound': vader_scores['compound'],
        'pos': vader_scores['pos'],
        'neg': vader_scores['neg'],
        'neu': vader_scores['neu'],
        'polarity': blob.sentiment.polarity,
        'subjectivity': blob.sentiment.subjectivity,
        'imperatives': imperatives / len(words) if words else 0,
        'adjectives': adjectives / len(words) if words else 0,
        'confidence': min(confidence, 1.0)  # Cap confidence at 1.0
    }

def get_emotion_category_enhanced(sentiment):
    compound = sentiment['compound']
    subjectivity = sentiment['subjectivity']
    neg = sentiment['neg']
    
    if compound <= -0.6 and neg >= 0.5:
        return 'angry'
    elif compound >= 0.5:
        return 'joyful'
    elif compound <= -0.5:
        return 'fearful'
    elif compound > 0 and subjectivity < 0.4:
        return 'peaceful'
    elif compound < 0 and subjectivity > 0.6:
        return 'remorseful'
    else:
        return 'reflective'

# Move the process_verses function definition before its usage
def process_verses(df):
    # Apply enhanced sentiment analysis
    sentiments = df['text'].apply(analyze_sentiment_enhanced)
    
    # Extract all sentiment features
    for feature in ['compound', 'pos', 'neg', 'neu', 'polarity', 'subjectivity', 'imperatives', 'adjectives', 'confidence']:
        df[feature] = sentiments.apply(lambda x: x[feature])
    
    # Apply enhanced emotion categorization
    df['emotion'] = sentiments.apply(get_emotion_category_enhanced)
    return df

# Load and process data
df = load_quran_data()
df = process_verses(df)

# Update recommendation function
def recommend_verses_enhanced(emotion_query, df, n_recommendations=5):
    # Define emotional centers with all features
    emotion_centers = {
        'guidance': {'compound': 0.2, 'subjectivity': 0.3, 'imperatives': 0.2},
        'joyful': {'compound': 0.6, 'subjectivity': 0.7, 'imperatives': 0.1},
        'angry': {'compound': -0.6, 'subjectivity': 0.8, 'imperatives': 0.2},
        'fearful': {'compound': -0.4, 'subjectivity': 0.7, 'imperatives': 0.15},
        'peaceful': {'compound': 0.3, 'subjectivity': 0.3, 'imperatives': 0.1},
        'remorseful': {'compound': -0.3, 'subjectivity': 0.4, 'imperatives': 0.1},
        'reflective': {'compound': 0.0, 'subjectivity': 0.5, 'imperatives': 0.1}
    }
    
    center = emotion_centers[emotion_query]
    
    # Calculate weighted similarity score
    df['similarity'] = np.sqrt(
        3 * (df['compound'] - center['compound'])**2 +
        2 * (df['subjectivity'] - center['subjectivity'])**2 +
        (df['imperatives'] - center['imperatives'])**2
    )
    
    return df.nsmallest(n_recommendations, 'similarity')[['surah_name', 'ayah_no', 'text', 'emotion', 'confidence']]

# Print distribution of emotions
print("\nDistribution of emotions in verses:")
print(df['emotion'].value_counts())

# Example recommendations for each emotion
for emotion in ['joyful', 'peaceful', 'angry', 'fearful', 'remorseful', 'reflective']:
    print(f"\nExample Recommendations for '{emotion}' emotion:")
    print(recommend_verses_enhanced(emotion, df))