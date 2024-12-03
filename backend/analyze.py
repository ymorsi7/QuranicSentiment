import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

class QuranAnalyzer:
    def __init__(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
        # Extract verses and flatten data
        self.verses = []
        self.process_data(self.data['children'])
        
        # Initialize TF-IDF
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),  # Consider both unigrams and bigrams
            max_features=5000
        )
        self.text_features = self.vectorizer.fit_transform([v['text'] for v in self.verses])
        
        # Normalize sentiment values to 0-1 range
        self.sentiment_scaler = MinMaxScaler()
        self.sentiment_values = self.sentiment_scaler.fit_transform(
            np.array([v['sentiment'] for v in self.verses]).reshape(-1, 1)
        )

    def process_data(self, surahs):
        """Process and flatten the hierarchical Quran data"""
        for surah in surahs:
            for verse in surah['children']:
                self.verses.append({
                    'surah_name': surah['name'],
                    'text': verse['ayah_en'],
                    'sentiment': verse['value'],
                    'ayah_no': verse['ayah_no_surah']
                })

    def analyze_sentiments(self):
        """Analyze sentiment distribution"""
        sentiments = [v['sentiment'] for v in self.verses]
        
        plt.figure(figsize=(10, 6))
        sns.histplot(sentiments, bins=30)
        plt.title('Distribution of Sentiment Values in Quran Verses')
        plt.xlabel('Sentiment Value')
        plt.ylabel('Count')
        plt.savefig('sentiment_distribution.png')
        plt.close()
        
        return {
            'mean': np.mean(sentiments),
            'median': np.median(sentiments),
            'std': np.std(sentiments),
            'min': np.min(sentiments),
            'max': np.max(sentiments)
        }

    def find_verses_by_sentiment(self, target_sentiment, n=5):
        """Find verses with similar sentiment values"""
        sentiments = np.array([v['sentiment'] for v in self.verses])
        closest_indices = np.argsort(np.abs(sentiments - target_sentiment))[:n]
        
        return [self.verses[i] for i in closest_indices]

    def find_similar_verses(self, verse_text, n=5):
        """Find verses with similar content"""
        verse_vector = self.vectorizer.transform([verse_text])
        similarities = cosine_similarity(verse_vector, self.text_features).flatten()
        top_indices = similarities.argsort()[-n:][::-1]
        
        return [
            {**self.verses[i], 'similarity': similarities[i]} 
            for i in top_indices
        ]

    def recommend_verses(self, mood_keywords, target_sentiment, n=5):
        """Recommend verses based on both content and sentiment"""
        # Get content similarity
        query_vector = self.vectorizer.transform([mood_keywords])
        content_similarities = cosine_similarity(query_vector, self.text_features).flatten()
        
        # Get sentiment similarity
        sentiment_similarities = 1 - np.abs(
            self.sentiment_values - self.sentiment_scaler.transform([[target_sentiment]])
        ).flatten()
        
        # Combine scores (weighted average)
        final_scores = (0.7 * content_similarities) + (0.3 * sentiment_similarities)
        top_indices = final_scores.argsort()[-n:][::-1]
        
        recommendations = []
        for idx in top_indices:
            recommendations.append({
                **self.verses[idx],
                'relevance_score': final_scores[idx]
            })
        
        return recommendations