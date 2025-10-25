import csv
import pickle
import re
import math
from collections import Counter

class PerfectSuicideDetector:
    def __init__(self):
        self.word_scores = {}
        
        self.suicide_keywords = {
            'suicide': 15.0, 'kill': 12.0, 'die': 11.0, 'death': 10.0, 'end': 8.0,
            'hurt': 6.0, 'pain': 6.0, 'depressed': 11.0, 'hopeless': 13.0,
            'worthless': 12.0, 'alone': 6.0, 'empty': 7.0, 'tired': 5.0,
            'give': 6.0, 'over': 5.0, 'done': 7.0, 'finished': 8.0,
            'cant': 6.0, 'anymore': 11.0, 'enough': 7.0, 'stop': 6.0,
            'escape': 8.0, 'gone': 7.0, 'disappear': 9.0, 'nothing': 6.0,
            'hate': 7.0, 'myself': 8.0, 'burden': 12.0, 'failure': 9.0,
            'goodbye': 13.0, 'farewell': 13.0, 'final': 9.0, 'last': 8.0
        }
        
        self.positive_keywords = {
            'happy': -10.0, 'joy': -11.0, 'love': -9.0, 'good': -7.0, 'great': -8.0,
            'amazing': -10.0, 'wonderful': -10.0, 'excited': -9.0, 'fun': -9.0,
            'beautiful': -9.0, 'hope': -11.0, 'future': -8.0, 'dream': -8.0,
            'success': -9.0, 'achievement': -9.0, 'proud': -9.0, 'grateful': -10.0,
            'blessed': -10.0, 'lucky': -9.0, 'positive': -9.0, 'optimistic': -10.0,
            'smile': -9.0, 'laugh': -10.0, 'celebrate': -10.0, 'enjoy': -9.0,
            'perfect': -9.0, 'excellent': -9.0, 'fantastic': -10.0, 'awesome': -10.0,
            'brilliant': -10.0, 'outstanding': -10.0, 'superb': -10.0, 'marvelous': -10.0,
            'delighted': -10.0, 'thrilled': -10.0, 'ecstatic': -11.0, 'elated': -11.0
        }
        
        self.neutral_strong = {
            'work': -4.0, 'job': -4.0, 'school': -4.0, 'study': -4.0, 'learn': -4.0,
            'family': -4.0, 'friend': -4.0, 'home': -4.0, 'food': -4.0, 'eat': -4.0,
            'sleep': -6.0, 'slept': -6.0, 'hours': -5.0, 'days': -3.0, 'recent': -4.0, 'watch': -4.0, 'read': -4.0, 'play': -4.0, 'music': -4.0,
            'movie': -4.0, 'book': -4.0, 'game': -4.0, 'sport': -4.0, 'travel': -5.0,
            'shopping': -5.0, 'cooking': -5.0, 'exercise': -5.0, 'walking': -5.0,
            'running': -5.0, 'swimming': -5.0, 'dancing': -5.0, 'singing': -5.0,
            'writing': -4.0, 'drawing': -4.0, 'painting': -4.0, 'photography': -4.0,
            'business': -4.0, 'meeting': -4.0, 'project': -4.0, 'planning': -4.0,
            'vacation': -6.0, 'holiday': -6.0, 'party': -6.0, 'wedding': -6.0,
            'birthday': -6.0, 'anniversary': -6.0, 'graduation': -6.0, 'promotion': -6.0,
            'money': -3.0, 'buy': -3.0, 'purchase': -3.0, 'sale': -3.0, 'discount': -4.0,
            'weather': -3.0, 'sunny': -4.0, 'warm': -4.0, 'cool': -3.0, 'nice': -4.0,
            'today': -2.0, 'tomorrow': -2.0, 'weekend': -4.0, 'morning': -3.0, 'evening': -3.0
        }
        
    def preprocess_text(self, text):
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        words = text.split()
        return [word for word in words if len(word) > 1]
    
    def train(self, texts, labels):
        print("Training perfect model...")
        
        suicide_texts = []
        non_suicide_texts = []
        
        for text, label in zip(texts, labels):
            words = self.preprocess_text(text)
            if label == 'suicide':
                suicide_texts.extend(words * 2)
            else:
                non_suicide_texts.extend(words * 4)  # Strong non-suicide boost
        
        suicide_counts = Counter(suicide_texts)
        non_suicide_counts = Counter(non_suicide_texts)
        
        total_suicide = len(suicide_texts)
        total_non_suicide = len(non_suicide_texts)
        
        all_words = set(suicide_counts.keys()) | set(non_suicide_counts.keys())
        
        for word in all_words:
            suicide_freq = suicide_counts.get(word, 0) / total_suicide
            non_suicide_freq = non_suicide_counts.get(word, 0) / total_non_suicide
            
            if suicide_freq > 0 and non_suicide_freq > 0:
                score = math.log(suicide_freq / non_suicide_freq)
            elif suicide_freq > 0:
                score = 1.8
            else:
                score = -3.0  # Very strong non-suicide indicator
                
            self.word_scores[word] = score
        
        print(f"Training completed on {len(all_words)} words")
    
    def predict(self, text):
        words = self.preprocess_text(text)
        
        if not words:
            return 'non-suicide', 0.1
        
        ml_score = 0
        keyword_score = 0
        word_count = 0
        
        for word in words:
            if word in self.word_scores:
                ml_score += self.word_scores[word]
                word_count += 1
            
            if word in self.suicide_keywords:
                keyword_score += self.suicide_keywords[word]
            elif word in self.positive_keywords:
                keyword_score += self.positive_keywords[word]
            elif word in self.neutral_strong:
                keyword_score += self.neutral_strong[word]
        
        if word_count > 0:
            avg_ml_score = ml_score / word_count
        else:
            avg_ml_score = 0
        
        # Text length bonus for confidence
        length_factor = min(1.2, len(words) / 10)
        final_score = (avg_ml_score + (keyword_score * 0.045)) * length_factor
        
        probability = 1 / (1 + math.exp(-final_score))
        
        # Smart confidence based on context
        abs_score = abs(final_score)
        
        if probability >= 0.70 or keyword_score > 20:
            confidence = min(0.95, 0.80 + abs_score * 0.05)
            return 'suicide', confidence
        elif probability >= 0.55:
            confidence = min(0.85, 0.70 + abs_score * 0.03)
            return 'suicide', confidence
        elif probability <= 0.30 or keyword_score < -15:
            confidence = min(0.92, 0.75 + abs_score * 0.04)
            return 'non-suicide', confidence
        elif probability <= 0.45:
            confidence = min(0.88, 0.68 + abs_score * 0.04)
            return 'non-suicide', confidence
        else:
            # Borderline cases
            confidence = 0.62
            return 'non-suicide', confidence

# Train and test
texts = []
labels = []

with open('Suicide_Detection.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        if i >= 15000:
            break
        texts.append(row['text'])
        labels.append(row['class'])

model = PerfectSuicideDetector()
model.train(texts[:10000], labels[:10000])

# Test
correct = 0
suicide_found = 0
total_suicide = 0
non_suicide_correct = 0
total_non_suicide = 0

for i in range(10000, 15000):
    pred, conf = model.predict(texts[i])
    actual = labels[i]
    
    if pred == actual:
        correct += 1
    
    if actual == 'suicide':
        total_suicide += 1
        if pred == 'suicide':
            suicide_found += 1
    else:
        total_non_suicide += 1
        if pred == 'non-suicide':
            non_suicide_correct += 1

accuracy = (correct / 5000) * 100
suicide_rate = (suicide_found / total_suicide) * 100 if total_suicide > 0 else 0
non_suicide_rate = (non_suicide_correct / total_non_suicide) * 100 if total_non_suicide > 0 else 0

print(f"\nPERFECT MODEL RESULTS:")
print(f"ACCURACY: {accuracy:.1f}%")
print(f"SUICIDE DETECTION: {suicide_rate:.1f}%")
print(f"NON-SUICIDE DETECTION: {non_suicide_rate:.1f}%")

with open('perfect_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Perfect model saved!")