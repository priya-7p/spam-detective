#!/usr/bin/env python3
"""
ML Spam Classifier Backend
A complete implementation using scikit-learn for actual ML processing
"""

import re
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import logging

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    print("Warning: Could not download NLTK data")

app = Flask(__name__)
CORS(app)

class SpamClassifier:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.is_trained = False
        self.metrics = {}
        
    def preprocess_text(self, text):
        """Preprocess text for classification"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        try:
            tokens = word_tokenize(text)
            # Remove stopwords
            stop_words = set(stopwords.words('english'))
            tokens = [token for token in tokens if token not in stop_words]
        except:
            # Fallback if NLTK data is not available
            tokens = text.split()
            # Basic stopwords
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            tokens = [token for token in tokens if token not in stop_words]
        
        return ' '.join(tokens)
    
    def extract_features(self, text):
        """Extract additional features from text"""
        features = {}
        
        # Count suspicious keywords
        spam_keywords = [
            'free', 'win', 'winner', 'congratulations', 'prize', 'money', 'cash',
            'urgent', 'limited time', 'act now', 'click here', 'guarantee',
            'no strings attached', 'exclusive', 'offer expires', 'claim now'
        ]
        
        suspicious_words = [word for word in spam_keywords if word in text.lower()]
        features['suspicious_word_count'] = len(suspicious_words)
        features['suspicious_words'] = suspicious_words
        
        # URL count
        urls = re.findall(r'https?://[^\s]+', text)
        features['url_count'] = len(urls)
        
        # Exclamation marks
        features['exclamation_count'] = text.count('!')
        
        # Capital letters percentage
        caps_count = sum(1 for c in text if c.isupper())
        features['caps_percentage'] = round((caps_count / len(text)) * 100) if text else 0
        
        return features
    
    def train_model(self, emails, labels, algorithm='naive_bayes'):
        """Train the spam classifier"""
        print("Preprocessing emails...")
        
        # Preprocess emails
        processed_emails = [self.preprocess_text(email) for email in emails]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            processed_emails, labels, test_size=0.2, random_state=42
        )
        
        print(f"Training with {len(X_train)} emails...")
        
        # Create pipeline
        if algorithm == 'naive_bayes':
            self.model = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
                ('classifier', MultinomialNB())
            ])
        else:  # SVM
            self.model = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
                ('classifier', SVC(probability=True, random_state=42))
            ])
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, pos_label='spam'),
            'recall': recall_score(y_test, y_pred, pos_label='spam'),
            'f1_score': f1_score(y_test, y_pred, pos_label='spam')
        }
        
        self.is_trained = True
        print("Model trained successfully!")
        print(f"Accuracy: {self.metrics['accuracy']:.3f}")
        print(f"Precision: {self.metrics['precision']:.3f}")
        print(f"Recall: {self.metrics['recall']:.3f}")
        print(f"F1-Score: {self.metrics['f1_score']:.3f}")
        
        return self.metrics
    
    def predict(self, text):
        """Predict if an email is spam or ham"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Get prediction and probability
        prediction = self.model.predict([processed_text])[0]
        probabilities = self.model.predict_proba([processed_text])[0]
        
        # Get confidence (probability of predicted class)
        if prediction == 'spam':
            confidence = probabilities[1]  # Assuming spam is class 1
        else:
            confidence = probabilities[0]  # Assuming ham is class 0
        
        # Extract additional features
        features = self.extract_features(text)
        
        return {
            'prediction': prediction,
            'confidence': float(confidence),
            'features': features
        }
    
    def save_model(self, filepath):
        """Save trained model to file"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'metrics': self.metrics
            }, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model from file"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.metrics = data['metrics']
                self.is_trained = True
            print(f"Model loaded from {filepath}")
            return True
        except FileNotFoundError:
            print(f"Model file {filepath} not found")
            return False

# Initialize classifier
classifier = SpamClassifier()

# Sample training data (you would replace this with a real dataset)
def create_sample_dataset():
    """Create a sample dataset for demonstration"""
    spam_emails = [
        "URGENT! Claim your $1000 prize now! Click here to win big money!",
        "Congratulations! You've won a free vacation! Act now, limited time offer!",
        "Get rich quick! Make money fast with our exclusive system!",
        "Free money waiting for you! No strings attached, claim now!",
        "Winner! You've been selected for our cash prize! Click to claim!",
    ] * 100  # Repeat to make dataset larger
    
    ham_emails = [
        "Hi, let's schedule our meeting for tomorrow at 2 PM in the conference room.",
        "Your order has been shipped and will arrive within 3-5 business days.",
        "Thank you for your subscription. Here's your monthly newsletter.",
        "Reminder: Your appointment is scheduled for next Monday at 10 AM.",
        "The quarterly report is ready for review. Please find it attached.",
    ] * 100  # Repeat to make dataset larger
    
    emails = spam_emails + ham_emails
    labels = ['spam'] * len(spam_emails) + ['ham'] * len(ham_emails)
    
    return emails, labels

@app.route('/train', methods=['POST'])
def train_model():
    """Train the spam classifier"""
    try:
        data = request.json
        algorithm = data.get('algorithm', 'naive_bayes')
        
        # Create sample dataset (in real application, you'd load from file/database)
        emails, labels = create_sample_dataset()
        
        # Train model
        metrics = classifier.train_model(emails, labels, algorithm)
        
        # Save model
        classifier.save_model('spam_classifier.pkl')
        
        return jsonify({
            'success': True,
            'metrics': {k: float(v) for k, v in metrics.items()},
            'message': 'Model trained successfully'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/predict', methods=['POST'])
def predict_email():
    """Classify an email as spam or ham"""
    try:
        data = request.json
        email_text = data.get('text', '')
        
        if not email_text:
            return jsonify({
                'success': False,
                'error': 'No email text provided'
            }), 400
        
        # Try to load model if not trained
        if not classifier.is_trained:
            if not classifier.load_model('spam_classifier.pkl'):
                # Train with sample data if no model exists
                emails, labels = create_sample_dataset()
                classifier.train_model(emails, labels)
                classifier.save_model('spam_classifier.pkl')
        
        # Make prediction
        result = classifier.predict(email_text)
        
        return jsonify({
            'success': True,
            'result': result
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Get model performance metrics"""
    try:
        if not classifier.is_trained:
            if not classifier.load_model('spam_classifier.pkl'):
                return jsonify({
                    'success': False,
                    'error': 'No trained model available'
                }), 404
        
        return jsonify({
            'success': True,
            'metrics': {k: float(v) for k, v in classifier.metrics.items()}
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'success': True,
        'status': 'healthy',
        'model_trained': classifier.is_trained
    })

if __name__ == '__main__':
    print("Starting ML Spam Classifier Backend...")
    print("Available endpoints:")
    print("  POST /train - Train the model")
    print("  POST /predict - Classify email")
    print("  GET /metrics - Get model metrics")
    print("  GET /health - Health check")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
