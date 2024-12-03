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
    
    # Enhanced POS tag analysis with weights
    linguistic_features = {
        'imperatives': sum(2 for _, tag in pos_tags if tag.startswith('VB')),
        'adjectives': sum(1.5 for _, tag in pos_tags if tag.startswith('JJ')),
        'adverbs': sum(1.2 for _, tag in pos_tags if tag.startswith('RB')),
        'nouns': sum(0.8 for _, tag in pos_tags if tag.startswith('NN'))
    }
    
    # Expanded emotional word lists specific to Quranic context
    emotional_words = {
        'joyful': [
            'paradise', 'reward', 'blessed', 'mercy', 'glad', 'rejoice', 'happy', 'delight', 
            'pleasure', 'bliss', 'joy', 'garden', 'success', 'triumph', 'victory'
        ],
        'peaceful': [
            'peace', 'tranquil', 'calm', 'gentle', 'harmony', 'serene', 'quiet', 'still',
            'rest', 'ease', 'comfort', 'secure', 'safe', 'protect', 'guide'
        ],
        'fearful': [
            'fear', 'punishment', 'warning', 'terrible', 'severe', 'doom', 'torment', 
            'horror', 'dread', 'terror', 'frightful', 'awful', 'painful', 'suffering'
        ],
        'angry': [
            'wrath', 'curse', 'condemn', 'destroy', 'vengeance', 'fury', 'rage', 
            'anger', 'punishment', 'disgrace', 'humiliate', 'shame', 'painful'
        ],
        'remorseful': [
            'forgive', 'repent', 'sorry', 'regret', 'mercy', 'pardon', 'forgiveness',
            'guilt', 'shame', 'wrong', 'sin', 'mistake', 'error', 'return'
        ],
        'reflective': [
            'ponder', 'think', 'reflect', 'contemplate', 'consider', 'remember',
            'understand', 'wisdom', 'sign', 'lesson', 'example', 'learn', 'know'
        ]
    }
    
    # Calculate emotional scores with context
    emotional_scores = {}
    lower_text = text.lower()
    for emotion, words_list in emotional_words.items():
        # Count exact matches and partial matches
        exact_matches = sum(2 for word in words_list if f" {word} " in f" {lower_text} ")
        partial_matches = sum(1 for word in words_list if word in lower_text)
        total_score = (exact_matches + partial_matches) / (len(words_list) * 2)
        emotional_scores[emotion] = min(1.0, total_score * 1.5)  # Scale up but cap at 1.0
    
    # Enhanced confidence calculation
    base_confidence = abs(vader_scores['compound'])
    emotional_intensity = max(emotional_scores.values()) if emotional_scores else 0
    linguistic_intensity = sum(linguistic_features.values()) / (len(words) + 1)
    
    # Weighted confidence calculation
    confidence = (
        base_confidence * 0.25 +  # VADER sentiment
        (blob.sentiment.subjectivity * 0.15) +  # TextBlob subjectivity
        (emotional_intensity * 0.4) +  # Emotional keyword matching (increased weight)
        min(1.0, linguistic_intensity) * 0.2  # Linguistic features
    )
    
    # Apply enhanced sigmoid scaling
    confidence = 1 / (1 + np.exp(-6 * (confidence - 0.3)))  # Adjusted parameters
    
    return {
        'compound': vader_scores['compound'],
        'pos': vader_scores['pos'],
        'neg': vader_scores['neg'],
        'neu': vader_scores['neu'],
        'polarity': blob.sentiment.polarity,
        'subjectivity': blob.sentiment.subjectivity,
        'imperatives': linguistic_features['imperatives'] / (len(words) + 1),
        'adjectives': linguistic_features['adjectives'] / (len(words) + 1),
        'emotional_scores': emotional_scores,
        'confidence': confidence
    }

def get_emotion_category_enhanced(sentiment):
    compound = sentiment['compound']
    subjectivity = sentiment['subjectivity']
    emotional_scores = sentiment['emotional_scores']
    
    # Get the dominant emotion from keyword analysis
    dominant_emotion = max(emotional_scores.items(), key=lambda x: x[1])[0] if emotional_scores else None
    
    # Combine VADER sentiment with keyword analysis
    if abs(compound) >= 0.5:  # Strong sentiment
        if compound >= 0.5 and emotional_scores.get('joyful', 0) > 0:
            return 'joyful'
        elif compound <= -0.5 and emotional_scores.get('angry', 0) > 0:
            return 'angry'
        elif compound <= -0.5 and emotional_scores.get('fearful', 0) > 0:
            return 'fearful'
    
    # Use emotional scores for more nuanced classification
    if dominant_emotion and emotional_scores[dominant_emotion] > 0.3:
        return dominant_emotion
    
    # Default classifications based on compound and subjectivity
    if compound > 0 and subjectivity < 0.4:
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
    for feature in ['compound', 'pos', 'neg', 'neu', 'polarity', 'subjectivity', 'imperatives', 'adjectives', 'emotional_scores', 'confidence']:
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