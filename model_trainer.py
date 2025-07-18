# This script trains your model and saves the required .pkl files
import pandas as pd
import numpy as np
import pickle
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def wordopt(text):
    """Text preprocessing function"""
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"\\W", " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

def train_and_save_model():
    # Load datasets from your datasets.zip
    data_fake = pd.read_csv('Fake.csv')
    data_true = pd.read_csv('True.csv')
    
    # Add labels
    data_fake["class"] = 0
    data_true['class'] = 1
    
    # Merge and prepare data
    data_merge = pd.concat([data_fake, data_true], axis=0)
    data = data_merge.drop(['title', 'subject', 'date'], axis=1)
    data = data.sample(frac=1).reset_index(drop=True)
    
    # Preprocess text
    data['text'] = data['text'].apply(wordopt)
    
    # Split data
    X = data['text']
    y = data['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Create TF-IDF vectorizer
    vectorization = TfidfVectorizer(max_features=10000)
    X_train_tfidf = vectorization.fit_transform(X_train)
    X_test_tfidf = vectorization.transform(X_test)
    
    # Train best model (Logistic Regression typically performs best)
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_tfidf, y_train)
    
    # Save model and vectorizer
    with open('model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    
    with open('tfidf.pkl', 'wb') as vectorizer_file:
        pickle.dump(vectorization, vectorizer_file)
    
    print("✅ Model and vectorizer saved successfully!")

if __name__ == "__main__":
    train_and_save_model()
