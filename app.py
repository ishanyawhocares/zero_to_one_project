import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime
from predict import load_model_and_tokenizer, predict_sentiment # Import from your file

# --- 1. INITIALIZATION ---
print("Starting application...")

# Load environment variables from .env file
load_dotenv()

# Initialize Flask App
app = Flask(__name__, template_folder='templates')
CORS(app) # Allows requests from your frontend

# --- 2. LOAD ML MODEL (Done only once on startup) ---
MODEL_PATH = 'sentiment_nn_model.h5'
TOKENIZER_PATH = 'nn_tokenizer.joblib'
# IMPORTANT: This mapping must match the one used during training
LABEL_MAPPING = {0: 'Bad', 1: 'Neutral', 2: 'Good'} # Correct mapping

# Load the model and tokenizer into memory
loaded_model, loaded_tokenizer = load_model_and_tokenizer(MODEL_PATH, TOKENIZER_PATH)
if loaded_model is None:
    print("FATAL ERROR: Could not load ML model. Exiting.")
    exit()

# --- 3. DATABASE CONNECTION ---
MONGO_URI = os.getenv('MONGO_URI')
if not MONGO_URI:
    print("FATAL ERROR: MONGO_URI environment variable not set.")
    exit()

client = MongoClient(MONGO_URI)
db = client.sentiment_analyzer_db # Your database name
reviews_collection = db.reviews # Your collection name
print("Successfully connected to MongoDB.")

# --- 4. DEFINE API ROUTES ---

# Route to serve the main HTML page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle sentiment analysis requests
@app.route('/analyze', methods=['POST'])
def analyze():
    # Get data from the frontend's request
    data = request.get_json()
    movie_title = data.get('movieTitle')
    review_text = data.get('movieReview')

    if not movie_title or not review_text:
        return jsonify({"error": "Movie title and review are required."}), 400

    # Get sentiment prediction from your model
    sentiment, confidence = predict_sentiment(loaded_model, loaded_tokenizer, review_text, LABEL_MAPPING)

    # Prepare data for database
    review_document = {
        "movie_title": movie_title,
        "review_text": review_text,
        "sentiment": sentiment,
        "confidence": float(confidence), # Convert numpy float to python float
        "timestamp": datetime.utcnow()
    }

    # Save to MongoDB
    try:
        reviews_collection.insert_one(review_document)
        print(f"Successfully saved review for '{movie_title}' to database.")
    except Exception as e:
        print(f"Error saving to database: {e}")
        # We can still return the result to the user even if DB save fails
        pass

    # Emojis for the frontend
    emoji_map = {
        'Good': '😍',
        'Bad': '😞',
        'Neutral': '😐'
    }

    # Return the result to the frontend
    return jsonify({
        "movieTitle": movie_title,
        "sentiment": sentiment,
        "confidence": f"{confidence:.2%}",
        "emoji": emoji_map.get(sentiment, '🤔'),
        "explanation": f"Our AI model analyzed your review and classified it as {sentiment} with {confidence:.0%} confidence."
    })

# --- 5. RUN THE APP ---
if __name__ == '__main__':
    # Use port 5000 for development
    app.run(host='0.0.0.0', port=5000, debug=True)