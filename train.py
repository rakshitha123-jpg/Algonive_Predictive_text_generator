import os
import sys
sys.path.append('src')

from src.ngram_model import NGramModel
from src.context_predictor import CustomDictionary

def create_sample_dataset():
    sample_text = '''
    The quick brown fox jumps over the lazy dog. The fox is quick and agile.
    Artificial intelligence is transforming the world. Machine learning models
    can predict text based on patterns. Natural language processing helps computers
    understand human language. Text prediction is useful for autocomplete features.
    Deep learning neural networks improve text generation. Language models are
    becoming more sophisticated. Transformers revolutionized NLP applications.
    BERT and GPT are popular language models. Text completion saves time.
    Smart keyboards use predictive text. Mobile phones benefit from text prediction.
    Word suggestion improves typing efficiency. Context-aware predictions are more accurate.
    Personalized dictionaries enhance user experience. Custom words improve prediction accuracy.
    '''
    
    with open('data/sample_text.txt', 'w') as f:
        f.write(sample_text)
    
    return sample_text

def main():
    print('Creating sample dataset...')
    sample_text = create_sample_dataset()
    
    print('Training N-gram models...')
    
    # Train bigram model
    bigram_model = NGramModel(n=2)
    bigram_model.train(sample_text)
    bigram_model.save_model('models/bigram_model.pkl')
    
    # Train trigram model
    trigram_model = NGramModel(n=3)
    trigram_model.train(sample_text)
    trigram_model.save_model('models/trigram_model.pkl')
    
    print('Creating custom dictionary...')
    custom_dict = CustomDictionary()
    
    # Add some custom words
    custom_words = [
        ('python', 10), ('javascript', 8), ('react', 7), ('nodejs', 6),
        ('tensorflow', 5), ('pytorch', 5), ('github', 9), ('vscode', 8),
        ('docker', 6), ('kubernetes', 4), ('aws', 7), ('azure', 5)
    ]
    
    for word, freq in custom_words:
        custom_dict.add_word(word, freq)
    
    custom_dict.save_dictionary()
    
    print('Training completed!')
    print('Models saved:')
    print('- Bigram model: models/bigram_model.pkl')
    print('- Trigram model: models/trigram_model.pkl')
    print('- Custom dictionary: data/custom_dictionary.json')
    
    # Test predictions
    print('')
    print('Testing predictions...')
    test_contexts = ['the', 'machine learning', 'artificial intelligence', 'text']
    
    for context in test_contexts:
        predictions = bigram_model.predict_next(context.split(), top_k=3)
        print('Context: ' + context + ' -> Predictions: ' + str(predictions))

if __name__ == '__main__':
    main()
