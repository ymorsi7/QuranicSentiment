from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/api/recommend', methods=['POST'])
def recommend():
    data = request.json
    emotion = data.get('emotion')
    intensity = data.get('intensity')
    context = data.get('context')
    
    # TODO: Implement your recommendation logic here
    # For now, returning dummy data
    dummy_response = [{
        "surah_name": "Example Surah",
        "ayah_no": 1,
        "text": "Example verse text",
        "relevance_score": 0.95,
        "sentiment": "positive"
    }]
    
    return jsonify(dummy_response)

if __name__ == '__main__':
    app.run(port=5000, debug=True)