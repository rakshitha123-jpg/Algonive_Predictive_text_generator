import json
from typing import List, Dict, Set, Tuple
from .ngram_model import NGramModel

class CustomDictionary:
    def __init__(self):
        self.custom_words: Set[str] = set()
        self.word_frequencies: Dict[str, int] = {}
        self.dictionary_file = 'data/custom_dictionary.json'
        
    def add_word(self, word: str, frequency: int = 1):
        word = word.lower().strip()
        self.custom_words.add(word)
        self.word_frequencies[word] = self.word_frequencies.get(word, 0) + frequency
        
    def remove_word(self, word: str):
        word = word.lower().strip()
        self.custom_words.discard(word)
        if word in self.word_frequencies:
            del self.word_frequencies[word]
    
    def get_frequency(self, word: str) -> int:
        return self.word_frequencies.get(word.lower().strip(), 0)
    
    def load_dictionary(self):
        try:
            with open(self.dictionary_file, 'r') as f:
                data = json.load(f)
                self.custom_words = set(data.get('words', []))
                self.word_frequencies = data.get('frequencies', {})
        except FileNotFoundError:
            pass
    
    def save_dictionary(self):
        data = {
            'words': list(self.custom_words),
            'frequencies': self.word_frequencies
        }
        with open(self.dictionary_file, 'w') as f:
            json.dump(data, f, indent=2)

class ContextAwarePredictor:
    def __init__(self, ngram_model: NGramModel, custom_dict: CustomDictionary):
        self.model = ngram_model
        self.dictionary = custom_dict
        self.context_window = 50
        
    def predict_with_context(self, text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        tokens = self.model.tokenize(text)
        
        if len(tokens) == 0:
            return self._get_fallback_predictions(top_k)
        
        # Get n-gram predictions
        ngram_predictions = self.model.predict_next(tokens, top_k)
        
        # Boost custom dictionary words
        boosted_predictions = []
        for word, prob in ngram_predictions:
            custom_freq = self.dictionary.get_frequency(word)
            if custom_freq > 0:
                # Boost probability for custom words
                boosted_prob = prob * (1 + custom_freq * 0.1)
                boosted_predictions.append((word, boosted_prob))
            else:
                boosted_predictions.append((word, prob))
        
        # Normalize probabilities
        total_prob = sum(p[1] for p in boosted_predictions)
        if total_prob > 0:
            boosted_predictions = [(w, p/total_prob) for w, p in boosted_predictions]
        
        return sorted(boosted_predictions, key=lambda x: x[1], reverse=True)[:top_k]
    
    def _get_fallback_predictions(self, top_k: int) -> List[Tuple[str, float]]:
        # Return most frequent custom words as fallback
        if not self.dictionary.word_frequencies:
            return []
        
        sorted_words = sorted(self.dictionary.word_frequencies.items(), 
                            key=lambda x: x[1], reverse=True)
        total_freq = sum(self.dictionary.word_frequencies.values())
        
        return [(word, freq/total_freq) for word, freq in sorted_words[:top_k]]
