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

def load_quran_data(file_path='quran.json'):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
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

def analyze_sentiment_enhanced(text, context='quranic'):
    vader = SentimentIntensityAnalyzer()
    vader_scores = vader.polarity_scores(text)
    blob = TextBlob(text)
    words = word_tokenize(text)
    pos_tags = nltk.pos_tag(words)
    linguistic_features = {
        'imperatives': sum(2 for _, tag in pos_tags if tag.startswith('VB')),
        'adjectives': sum(1.5 for _, tag in pos_tags if tag.startswith('JJ')),
        'adverbs': sum(1.2 for _, tag in pos_tags if tag.startswith('RB')),
        'nouns': sum(0.8 for _, tag in pos_tags if tag.startswith('NN'))
    }
    
    emotional_words = {
        'joyful': [
            # Spiritual joy
            'paradise', 'reward', 'blessed', 'mercy', 'glad', 'rejoice', 'happy', 'delight', 
            'pleasure', 'bliss', 'joy', 'garden', 'success', 'triumph', 'victory',
            'eternal', 'divine', 'gracious', 'abundant', 'prosper', 'flourish', 'radiant',
            'light', 'illuminated', 'grateful', 'thankful', 'praise', 'exalt', 'honor',
            'glorify', 'celebrate', 'jubilant', 'elated', 'overjoyed', 'ecstatic',
            
            # Rewards and blessings
            'bounty', 'gift', 'treasure', 'precious', 'valuable', 'worthy', 'noble',
            'elevated', 'honored', 'distinguished', 'favored', 'chosen', 'selected',
            'special', 'extraordinary', 'magnificent', 'splendid', 'glorious', 'wonderful',
            'marvelous', 'excellent', 'supreme', 'perfect', 'complete', 'fulfilled',
            
            # Positive outcomes
            'achieve', 'accomplish', 'attain', 'succeed', 'prosper', 'thrive', 'excel',
            'advance', 'progress', 'grow', 'develop', 'improve', 'enhance', 'strengthen',
            'empower', 'uplift', 'inspire', 'motivate', 'encourage', 'support'
        ],
        
        'peaceful': [
            # Inner peace
            'peace', 'tranquil', 'calm', 'gentle', 'harmony', 'serene', 'quiet', 'still',
            'rest', 'ease', 'comfort', 'secure', 'safe', 'protect', 'guide', 'balanced',
            'centered', 'grounded', 'steady', 'stable', 'composed', 'collected', 'peaceful',
            'placid', 'undisturbed', 'untroubled', 'content', 'satisfied', 'fulfilled',
            
            # Divine protection
            'shelter', 'refuge', 'sanctuary', 'haven', 'fortress', 'shield', 'guard',
            'preserve', 'defend', 'safeguard', 'secure', 'protect', 'watch', 'care',
            'nurture', 'support', 'sustain', 'maintain', 'uphold', 'strengthen',
            
            # Spiritual guidance
            'guide', 'direct', 'lead', 'show', 'teach', 'instruct', 'educate', 'enlighten',
            'illuminate', 'clarify', 'explain', 'reveal', 'demonstrate', 'indicate',
            'point', 'signal', 'beacon', 'light', 'path', 'way', 'journey', 'direction'
        ],
        
        'fearful': [
            # Divine punishment
            'fear', 'punishment', 'warning', 'terrible', 'severe', 'doom', 'torment',
            'horror', 'dread', 'terror', 'frightful', 'awful', 'painful', 'suffering',
            'anguish', 'agony', 'misery', 'distress', 'affliction', 'tribulation',
            
            # Consequences
            'judgment', 'reckoning', 'account', 'consequence', 'result', 'outcome',
            'effect', 'impact', 'repercussion', 'aftermath', 'penalty', 'retribution',
            'vengeance', 'revenge', 'payback', 'justice', 'desert', 'due', 'merit',
            
            # Spiritual fear
            'awe', 'reverence', 'respect', 'regard', 'esteem', 'veneration', 'deference',
            'submission', 'surrender', 'yield', 'bow', 'humble', 'lower', 'diminish',
            'reduce', 'lessen', 'decrease', 'minimize', 'shrink', 'contract'
        ],
        
        'angry': [
            # Divine wrath
            'wrath', 'curse', 'condemn', 'destroy', 'vengeance', 'fury', 'rage',
            'anger', 'punishment', 'disgrace', 'humiliate', 'shame', 'painful',
            'fierce', 'intense', 'severe', 'harsh', 'strict', 'stern', 'rigid',
            
            # Justice and retribution
            'justice', 'judgment', 'verdict', 'sentence', 'decree', 'pronouncement',
            'declaration', 'proclamation', 'announcement', 'statement', 'utterance',
            'command', 'order', 'directive', 'instruction', 'mandate', 'dictate',
            
            # Consequences of disobedience
            'disobey', 'rebel', 'revolt', 'resist', 'oppose', 'defy', 'challenge',
            'confront', 'face', 'meet', 'encounter', 'experience', 'undergo', 'suffer',
            'endure', 'bear', 'withstand', 'sustain', 'support', 'carry'
        ],
        
        'remorseful': [
            # Repentance
            'forgive', 'repent', 'sorry', 'regret', 'mercy', 'pardon', 'forgiveness',
            'guilt', 'shame', 'wrong', 'sin', 'mistake', 'error', 'return', 'remorse',
            'contrition', 'penitence', 'compunction', 'self-reproach', 'self-blame',
            
            # Seeking forgiveness
            'seek', 'ask', 'request', 'plead', 'beg', 'implore', 'beseech', 'entreat',
            'supplicate', 'petition', 'appeal', 'pray', 'worship', 'adore', 'revere',
            'venerate', 'honor', 'respect', 'esteem', 'regard',
            
            # Spiritual cleansing
            'purify', 'cleanse', 'clean', 'wash', 'rinse', 'clear', 'purge', 'expunge',
            'remove', 'eliminate', 'eradicate', 'destroy', 'abolish', 'annihilate',
            'obliterate', 'wipe', 'erase', 'delete', 'cancel', 'nullify'
        ],
        
        'reflective': [
            # Contemplation
            'ponder', 'think', 'reflect', 'contemplate', 'consider', 'remember',
            'understand', 'wisdom', 'sign', 'lesson', 'example', 'learn', 'know',
            'meditate', 'muse', 'ruminate', 'cogitate', 'deliberate', 'study',
            
            # Understanding
            'comprehend', 'grasp', 'apprehend', 'perceive', 'discern', 'distinguish',
            'differentiate', 'discriminate', 'separate', 'divide', 'classify',
            'categorize', 'organize', 'arrange', 'order', 'structure', 'systematic',
            
            # Spiritual growth
            'grow', 'develop', 'progress', 'advance', 'proceed', 'continue', 'persist',
            'persevere', 'endure', 'last', 'remain', 'stay', 'abide', 'dwell',
            'inhabit', 'occupy', 'possess', 'own', 'have', 'hold'
        ]
    }
    
    conversational_emotional_words = {
        'joyful': [
            'happy', 'excited', 'delighted', 'glad', 'cheerful', 'thrilled',
            'wonderful', 'fantastic', 'amazing', 'great', 'love', 'enjoy',
            'fun', 'pleased', 'satisfied', 'proud', 'blessed', 'grateful',
            'optimistic', 'positive', 'enthusiastic', 'energetic', 'good'
        ],
        'peaceful': [
            'calm', 'relaxed', 'peaceful', 'quiet', 'tranquil', 'serene',
            'content', 'comfortable', 'safe', 'secure', 'balanced', 'steady',
            'gentle', 'easy', 'settled', 'harmonious', 'okay', 'fine',
            'alright', 'well', 'rested', 'composed', 'stable'
        ],
        'fearful': [
            'scared', 'afraid', 'worried', 'anxious', 'nervous', 'terrified',
            'frightened', 'uneasy', 'stressed', 'panicked', 'overwhelmed',
            'helpless', 'insecure', 'concerned', 'uncertain', 'confused',
            'doubtful', 'vulnerable', 'threatened', 'paranoid'
        ],
        'angry': [
            'angry', 'mad', 'furious', 'annoyed', 'irritated', 'frustrated',
            'upset', 'outraged', 'hate', 'bitter', 'enraged', 'hostile',
            'offended', 'insulted', 'hurt', 'betrayed', 'disrespected',
            'unfair', 'wrong', 'terrible'
        ],
        'remorseful': [
            'sorry', 'regret', 'guilty', 'ashamed', 'apologetic', 'bad',
            'mistake', 'fault', 'disappointed', 'failed', 'messed up',
            'wish', 'apology', 'forgiveness', 'remorse', 'embarrassed',
            'wrong', 'blame', 'responsible'
        ],
        'reflective': [
            'think', 'wonder', 'consider', 'realize', 'understand', 'feel',
            'sense', 'believe', 'suppose', 'guess', 'maybe', 'perhaps',
            'question', 'searching', 'trying', 'processing', 'dealing',
            'coping', 'learning', 'growing'
        ]
    }
    
    # Choose appropriate dictionary based on context
    words_dict = emotional_words if context == 'quranic' else conversational_emotional_words
    
    emotional_scores = {}
    lower_text = text.lower()
    for emotion, words_list in words_dict.items():
        exact_matches = sum(2 for word in words_list if f" {word} " in f" {lower_text} ")
        partial_matches = sum(1 for word in words_list if word in lower_text)
        total_score = (exact_matches + partial_matches) / (len(words_list) * 2)
        emotional_scores[emotion] = min(1.0, total_score * 1.5)  # Scale up but cap at 1.0
    
    base_confidence = abs(vader_scores['compound'])
    emotional_intensity = max(emotional_scores.values()) if emotional_scores else 0
    linguistic_intensity = sum(linguistic_features.values()) / (len(words) + 1)
    
    confidence = (
        base_confidence * 0.25 + 
        (blob.sentiment.subjectivity * 0.15) +  
        (emotional_intensity * 0.4) + 
        min(1.0, linguistic_intensity) * 0.2 
    )
    
    confidence = 1 / (1 + np.exp(-6 * (confidence - 0.3))) 
    
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
    
    dominant_emotion = max(emotional_scores.items(), key=lambda x: x[1])[0] if emotional_scores else None
    
    if abs(compound) >= 0.5: 
        if compound >= 0.5 and emotional_scores.get('joyful', 0) > 0:
            return 'joyful'
        elif compound <= -0.5 and emotional_scores.get('angry', 0) > 0:
            return 'angry'
        elif compound <= -0.5 and emotional_scores.get('fearful', 0) > 0:
            return 'fearful'
    
    if dominant_emotion and emotional_scores[dominant_emotion] > 0.3:
        return dominant_emotion
    
    if compound > 0 and subjectivity < 0.4:
        return 'peaceful'
    elif compound < 0 and subjectivity > 0.6:
        return 'remorseful'
    else:
        return 'reflective'

def process_verses(df):
    sentiments = df['text'].apply(analyze_sentiment_enhanced)
    
    for feature in ['compound', 'pos', 'neg', 'neu', 'polarity', 'subjectivity', 'imperatives', 'adjectives', 'emotional_scores', 'confidence']:
        df[feature] = sentiments.apply(lambda x: x[feature])
    
    df['emotion'] = sentiments.apply(get_emotion_category_enhanced)
    return df

df = load_quran_data()
df = process_verses(df)

def recommend_verses_enhanced(emotion_query, df, n_recommendations=5):
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
    
    df['similarity'] = np.sqrt(
        3 * (df['compound'] - center['compound'])**2 +
        2 * (df['subjectivity'] - center['subjectivity'])**2 +
        (df['imperatives'] - center['imperatives'])**2
    )
    
    return df.nsmallest(n_recommendations, 'similarity')[['surah_name', 'ayah_no', 'text', 'emotion', 'confidence']]

print("\nDistribution of emotions in verses:")
print(df['emotion'].value_counts())

for emotion in ['joyful', 'peaceful', 'angry', 'fearful', 'remorseful', 'reflective']:
    print(f"\nExample Recommendations for '{emotion}' emotion:")
    print(recommend_verses_enhanced(emotion, df))