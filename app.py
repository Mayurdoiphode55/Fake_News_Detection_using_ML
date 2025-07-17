import streamlit as st
import pickle
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved model and vectorizer
# Use st.cache_resource to load them only once and improve performance
@st.cache_resource
def load_assets():
    """Loads the pre-trained model and TF-IDF vectorizer."""
    try:
        with open('model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('tfidf.pkl', 'rb') as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)
        return model, vectorizer
    except FileNotFoundError:
        st.error("Model or vectorizer file not found. Please ensure 'model.pkl' and 'tfidf.pkl' are in the same directory.")
        return None, None

# Function to preprocess the input text (matches the notebook)
def wordopt(text):
    """Cleans and preprocesses the input text."""
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Load the assets
model, vectorizer = load_assets()

# Set up the Streamlit app title and description
st.title("📰 Fake News Detector")
st.write(
    "Enter a news article or headline below to check if it's likely to be real or fake. "
    "The model was trained on a dataset of news articles to classify them."
)
st.markdown("---")

# Create a text area for user input
input_text = st.text_area("Enter the news text here:", height=200, placeholder="Paste your news article...")

# Create a button to trigger the prediction
if st.button("Analyze News"):
    if model and vectorizer:
        if input_text.strip() == "":
            st.warning("Please enter some text to analyze.")
        else:
            # Preprocess the input text
            cleaned_text = wordopt(input_text)
            
            # Vectorize the cleaned text
            vectorized_text = vectorizer.transform([cleaned_text])
            
            # Make a prediction
            prediction = model.predict(vectorized_text)
            
            # Display the result
            st.markdown("---")
            st.subheader("Analysis Result")
            if prediction[0] == 0:
                st.success("✅ This looks like REAL news.")
            else:
                st.error("🚨 Warning: This might be FAKE news.")
    else:
        st.info("App is not ready. Model assets could not be loaded.")
