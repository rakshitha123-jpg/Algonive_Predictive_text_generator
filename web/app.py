from flask import Flask, render_template, request, jsonify
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ngram_model import NGramModel

app = Flask(__name__)

# Load models
bigram_model = NGramModel(n=2)
trigram_model = NGramModel(n=3)

try:
    bigram_model.load_model('../models/bigram_model.pkl')
    trigram_model.load_model('../models/trigram_model.pkl')
    print('Models loaded successfully!')
except FileNotFoundError:
    print('Models not found. Please run train.py first.')
    exit(1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')
    model_type = data.get('model', 'bigram')
    top_k = data.get('top_k', 5)
    
    if not text.strip():
        return jsonify({'predictions': []})
    
    if model_type == 'trigram':
        model = trigram_model
        predictions = model.predict_next(text.split(), top_k)
    else:  # default to bigram
        model = bigram_model
        predictions = model.predict_next(text.split(), top_k)
    
    return jsonify({
        'predictions': [{'word': word, 'probability': prob} for word, prob in predictions]
    })

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    start_text = data.get('start_text', '')
    max_length = data.get('max_length', 20)
    model_type = data.get('model', 'bigram')
    
    if model_type == 'trigram':
        model = trigram_model
    else:
        model = bigram_model
    
    generated_text = model.generate_text(start_text, max_length)
    
    return jsonify({'generated_text': generated_text})

@app.route('/suggestions', methods=['POST'])
def suggestions():
    data = request.get_json()
    text = data.get('text', '')
    
    # Get suggestions from multiple sources
    bigram_preds = bigram_model.predict_next(text.split(), 3)
    trigram_preds = trigram_model.predict_next(text.split(), 3)
    
    # Combine and deduplicate
    all_suggestions = {}
    for word, prob in bigram_preds:
        all_suggestions[word] = {'probability': prob, 'source': 'bigram'}
    for word, prob in trigram_preds:
        if word in all_suggestions:
            all_suggestions[word]['probability'] = max(all_suggestions[word]['probability'], prob)
            all_suggestions[word]['source'] += ', trigram'
        else:
            all_suggestions[word] = {'probability': prob, 'source': 'trigram'}
    
    # Sort by probability
    sorted_suggestions = sorted(all_suggestions.items(), key=lambda x: x[1]['probability'], reverse=True)
    
    return jsonify({
        'suggestions': [{'word': word, **details} for word, details in sorted_suggestions[:10]]
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
