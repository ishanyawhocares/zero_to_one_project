import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import os

try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    print('Downloading NLTK stopwords...')
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except nltk.downloader.DownloadError:
    print('Downloading NLTK wordnet...')
    nltk.download('wordnet')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print('Downloading NLTK punkt...')
    nltk.download('punkt')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

MAX_WORDS = 10000
MAX_LEN = 100

def preprocess_text(text):
    """Clean and preprocess text data"""
    # Ensure text is a string
    text = str(text)
    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags specifically <br />
    text = re.sub(r'<br\s*/?>', ' ', text)

    # Remove special characters and digits, keeping only letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove extra whitespace
    text = ' '.join(text.split())

    # Tokenize and remove stopwords and lemmatize
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    return ' '.join(words)

def fit_and_save_tokenizer(texts, file_path):
    """Fits a Tokenizer and saves it to a file."""
    print(f'Fitting tokenizer with max_words=10000...')
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)
    print(f'Saving tokenizer to {file_path}...')
    joblib.dump(tokenizer, file_path)
    print('Tokenizer saved successfully.')
    return tokenizer

def load_tokenizer(file_path):
    """Loads a Tokenizer from a file."""
    print(f'Loading tokenizer from {file_path}...')
    tokenizer = joblib.load(file_path)
    print('Tokenizer loaded successfully.')
    return tokenizer

def tokenize_and_pad_text(texts, tokenizer):
    """Converts text to sequences and pads them."""
    print(f'Tokenizing and padding text to max_len=100...')
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    print('Tokenization and padding complete.')
    return padded_sequences
