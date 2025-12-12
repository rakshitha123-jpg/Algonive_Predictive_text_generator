import json
import pickle
import random
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional

class NGramModel:
    def __init__(self, n: int = 2):
        self.n = n
        self.ngrams = defaultdict(Counter)
        self.vocab = set()
        self.start_token = '<START>'
        self.end_token = '<END>'
        
    def tokenize(self, text: str) -> List[str]:
        text = text.lower().replace('\n', ' ').replace('\t', ' ')
        tokens = text.split()
        return [token.strip('.,!?;:\"()[]{}') for token in tokens if token.strip()]
    
    def generate_ngrams(self, tokens: List[str]) -> List[Tuple[str, ...]]:
        padded_tokens = [self.start_token] * (self.n - 1) + tokens + [self.end_token]
        ngrams = []
        for i in range(len(padded_tokens) - self.n + 1):
            ngram = tuple(padded_tokens[i:i+self.n])
            ngrams.append(ngram)
        return ngrams
    
    def train(self, text: str):
        tokens = self.tokenize(text)
        self.vocab.update(tokens)
        ngrams = self.generate_ngrams(tokens)
        
        for ngram in ngrams:
            context = ngram[:-1]
            target = ngram[-1]
            self.ngrams[context][target] += 1
    
    def predict_next(self, context: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        if len(context) < self.n - 1:
            context = [self.start_token] * (self.n - 1 - len(context)) + context
        
        context_tuple = tuple(context[-(self.n-1):])
        
        if context_tuple not in self.ngrams:
            return []
        
        predictions = self.ngrams[context_tuple]
        total_count = sum(predictions.values())
        
        return [(word, count/total_count) for word, count in predictions.most_common(top_k)]
    
    def generate_text(self, start_text: str = '', max_length: int = 20) -> str:
        if start_text:
            tokens = self.tokenize(start_text)
        else:
            tokens = []
        
        generated = tokens.copy()
        
        for _ in range(max_length):
            predictions = self.predict_next(tokens)
            if not predictions:
                break
            
            next_word = random.choices([p[0] for p in predictions], 
                                     weights=[p[1] for p in predictions])[0]
            generated.append(next_word)
            tokens.append(next_word)
            
            if next_word == self.end_token:
                break
        
        return ' '.join(generated)
    
    def save_model(self, filepath: str):
        model_data = {
            'n': self.n,
            'ngrams': dict(self.ngrams),
            'vocab': list(self.vocab),
            'start_token': self.start_token,
            'end_token': self.end_token
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.n = model_data['n']
        self.ngrams = defaultdict(Counter, model_data['ngrams'])
        self.vocab = set(model_data['vocab'])
        self.start_token = model_data['start_token']
        self.end_token = model_data['end_token']
