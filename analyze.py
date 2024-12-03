import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

class QuranAnalyzer:
    def __init__(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
        # Extract verses and flatten data
        self.verses = []
        self.process_data(self.data['children'])
        
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

    def recommend_verses(self, target_sentiment, n=5):
        """Recommend verses based on sentiment"""
        sentiments = np.array([v['sentiment'] for v in self.verses])
        closest_indices = np.argsort(np.abs(sentiments - target_sentiment))[:n]
        
        recommendations = []
        for idx in closest_indices:
            recommendations.append({
                **self.verses[idx],
                'relevance_score': 1 - np.abs(sentiments[idx] - target_sentiment)
            })
        
        return recommendations