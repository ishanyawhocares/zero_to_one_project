import tensorflow as tf
import numpy as np
import joblib
import os
from preprocessing import preprocess_text, tokenize_and_pad_text, MAX_LEN

def load_model_and_tokenizer(model_path, tokenizer_path):
    """Loads the trained Keras model and tokenizer from files."""
    try:
        print(f'Loading model from {model_path}...')
        model = tf.keras.models.load_model(model_path)
        print('Model loaded successfully.')

        print(f'Loading tokenizer from {tokenizer_path}...')
        tokenizer = joblib.load(tokenizer_path)
        print('Tokenizer loaded successfully.')

        return model, tokenizer
    except FileNotFoundError:
        print(f'Error: Model or tokenizer file not found.')
        print(f'Please ensure {model_path} and {tokenizer_path} exist.')
        return None, None
    except Exception as e:
        print(f'An error occurred while loading files: {e}')
        return None, None

def predict_sentiment(model, tokenizer, review_text, label_mapping):
    """
    Takes raw review text, preprocesses it, and returns the predicted sentiment and confidence.
    """
    # 1. Clean the input text
    cleaned_text = preprocess_text(review_text)

    # 2. Convert the cleaned text to a padded sequence
    # Note: tokenize_and_pad_text expects a list of texts
    padded_sequence = tokenize_and_pad_text([cleaned_text], tokenizer)

    # 3. Use the model to predict the probabilities for each class
    prediction_probabilities = model.predict(padded_sequence, verbose=0)[0]

    # 4. Find the index of the class with the highest probability
    predicted_class_index = np.argmax(prediction_probabilities)

    # 5. Get the corresponding sentiment label from the mapping
    # Ensure the label_mapping is correctly passed or defined
    sentiment = label_mapping.get(predicted_class_index, 'Unknown')

    # 6. Get the confidence score
    confidence = prediction_probabilities[predicted_class_index]

    return sentiment, confidence

if __name__ == "__main__":
    # --- Configuration ---
    # Define the file paths for the saved model and tokenizer
    MODEL_PATH = 'sentiment_nn_model.h5'
    TOKENIZER_PATH = 'nn_tokenizer.joblib'
    # Define the label mapping (MUST match the training mapping)
    # Use the label_encoder from the notebook to get the correct mapping
    LABEL_MAPPING = dict(zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_))
    # --- Load Model and Tokenizer ---
    loaded_model, loaded_tokenizer = load_model_and_tokenizer(MODEL_PATH, TOKENIZER_PATH)

    if loaded_model is None or loaded_tokenizer is None:
        print("Failed to load model or tokenizer. Exiting.")
    else:
        print("\n--- Sentiment Analysis Predictor ---")
        # Print the actual labels being used by the model
        print(f"Model loaded for sentiment prediction (Negative, Neutral, Positive).")

        # --- Interactive Prediction Loop ---
        while True:
            user_input = input("\nEnter a review (or type 'exit' to quit): ")

            if user_input.lower() == 'exit':
                print("Exiting predictor. Goodbye!")
                break

            if not user_input.strip():
                print("Please enter some text.")
                continue

            # Get the prediction
            sentiment, confidence = predict_sentiment(loaded_model, loaded_tokenizer, user_input, LABEL_MAPPING)

            # Display the results
            print("-" * 40)
            print(f"  Prediction: Neutral")
            print(f"  Confidence: {confidence:.2%}")
            print("-" * 40)
