import json
import pickle
from datetime import datetime
from perfect_model import PerfectSuicideDetector

class AdaptiveLearningSystem:
    def __init__(self):
        self.base_model = None
        self.learning_data = []
        self.confidence_threshold = 0.6
        self.improvement_counter = 0
        
    def load_base_model(self):
        with open('perfect_model.pkl', 'rb') as f:
            self.base_model = pickle.load(f)
        return self.base_model
    
    def collect_interaction(self, text, prediction, confidence, user_feedback=None):
        """Collect user interactions for future learning"""
        interaction = {
            'text': text,
            'prediction': prediction,
            'confidence': float(confidence),
            'user_feedback': user_feedback,
            'timestamp': datetime.now().isoformat(),
            'needs_review': confidence < self.confidence_threshold
        }
        
        self.learning_data.append(interaction)
        self.save_learning_data()
        
        # Check if we have enough data for improvement
        if len(self.learning_data) >= 100:
            self.trigger_improvement_analysis()
    
    def save_learning_data(self):
        """Save learning data to file"""
        with open('learning_interactions.json', 'w') as f:
            json.dump(self.learning_data, f, indent=2)
    
    def load_learning_data(self):
        """Load existing learning data"""
        try:
            with open('learning_interactions.json', 'r') as f:
                self.learning_data = json.load(f)
        except FileNotFoundError:
            self.learning_data = []
    
    def trigger_improvement_analysis(self):
        """Analyze collected data for potential improvements"""
        low_confidence_cases = [d for d in self.learning_data if d['needs_review']]
        
        if len(low_confidence_cases) >= 20:
            self.improvement_counter += 1
            print(f"🔄 Improvement Analysis #{self.improvement_counter}")
            print(f"📊 Analyzed {len(low_confidence_cases)} low-confidence cases")
            
            # Identify patterns in low-confidence predictions
            self.identify_improvement_patterns(low_confidence_cases)
    
    def identify_improvement_patterns(self, cases):
        """Identify patterns that could improve model"""
        word_patterns = {}
        
        for case in cases:
            words = case['text'].lower().split()
            for word in words:
                if len(word) > 3:
                    if word not in word_patterns:
                        word_patterns[word] = {'count': 0, 'predictions': []}
                    word_patterns[word]['count'] += 1
                    word_patterns[word]['predictions'].append(case['prediction'])
        
        # Find words that appear frequently in uncertain predictions
        improvement_suggestions = []
        for word, data in word_patterns.items():
            if data['count'] >= 3:
                improvement_suggestions.append({
                    'word': word,
                    'frequency': data['count'],
                    'suggestion': f"Consider adding '{word}' to keyword dictionary"
                })
        
        self.save_improvement_suggestions(improvement_suggestions)
    
    def save_improvement_suggestions(self, suggestions):
        """Save improvement suggestions for future model updates"""
        with open('model_improvements.json', 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'suggestions': suggestions,
                'total_interactions': len(self.learning_data),
                'improvement_round': self.improvement_counter
            }, f, indent=2)
        
        print(f"💡 Generated {len(suggestions)} improvement suggestions")
        print("📁 Saved to model_improvements.json")
    
    def collect_feedback(self, text, feedback_type, analysis_id):
        """Collect user feedback for specific analysis"""
        feedback_entry = {
            'text': text,
            'feedback': feedback_type,
            'analysis_id': analysis_id,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add to learning data
        self.learning_data.append(feedback_entry)
        self.save_learning_data()
        
        # Save separate feedback file
        try:
            with open('user_feedback.json', 'r') as f:
                feedback_data = json.load(f)
        except FileNotFoundError:
            feedback_data = []
        
        feedback_data.append(feedback_entry)
        
        with open('user_feedback.json', 'w') as f:
            json.dump(feedback_data, f, indent=2)
    
    def get_learning_stats(self):
        """Get current learning statistics"""
        if not self.learning_data:
            return {'total': 0, 'low_confidence': 0, 'improvements': 0}
        
        low_conf = len([d for d in self.learning_data if d['needs_review']])
        
        # Count feedback
        feedback_entries = [d for d in self.learning_data if 'feedback' in d]
        positive_feedback = len([d for d in feedback_entries if d.get('feedback') == 'positive'])
        
        return {
            'total_interactions': len(self.learning_data),
            'low_confidence_cases': low_conf,
            'improvement_rounds': self.improvement_counter,
            'learning_rate': f"{(low_conf/len(self.learning_data)*100):.1f}%" if self.learning_data else "0%",
            'total_feedback': len(feedback_entries),
            'positive_feedback': positive_feedback
        }

# Global learning system instance
learning_system = AdaptiveLearningSystem()
learning_system.load_learning_data()
try:
    learning_system.load_base_model()
except FileNotFoundError:
    print("Base model not found for learning system")