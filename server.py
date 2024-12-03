from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from analyze import QuranAnalyzer

app = Flask(__name__)
CORS(app)

# Load the Quran data
with open('quran.json', 'r', encoding='utf-8') as f:
    quran_data = json.load(f)

# Initialize the QuranAnalyzer
analyzer = QuranAnalyzer('quran.json')

@app.route('/api/recommend', methods=['POST'])
def recommend():
    data = request.json
    target_sentiment = data.get('sentiment')
    
    # Use the recommendation logic from the QuranAnalyzer
    recommendations = analyzer.recommend_verses(target_sentiment, n=5)

    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(port=5000, debug=True)