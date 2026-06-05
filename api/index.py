"""
Flask API for AI ChatBot — deployed on Vercel
"""

import json
import os
import random
from datetime import datetime

import numpy as np
import joblib
from flask import Flask, request, jsonify, send_from_directory


app = Flask(__name__, static_folder="../static")

# ============================================================================
# Load Model & Intents
# ============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
INTENTS_FILE = os.path.join(BASE_DIR, "intents.json")
CONFIDENCE_THRESHOLD = 0.25


def load_chatbot():
    """Load all model components."""
    classifier = joblib.load(os.path.join(MODEL_DIR, "intent_classifier.pkl"))
    vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

    with open(INTENTS_FILE, "r", encoding="utf-8") as f:
        intents_data = json.load(f)

    responses = {}
    for intent in intents_data["intents"]:
        responses[intent["tag"]] = intent["responses"]

    return classifier, vectorizer, label_encoder, responses


# Load model at startup
classifier, vectorizer, label_encoder, responses = load_chatbot()


# ============================================================================
# Routes
# ============================================================================

@app.route("/")
def home():
    """Serve the frontend HTML page."""
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    """Handle chat messages and return bot response."""
    data = request.get_json()

    if not data or "message" not in data:
        return jsonify({"error": "No message provided"}), 400

    user_input = data["message"].strip()

    if not user_input:
        return jsonify({"response": "I didn't catch that. Could you say something?"})

    # Predict intent
    processed_input = user_input.lower().strip()
    input_vector = vectorizer.transform([processed_input])
    probabilities = classifier.predict_proba(input_vector)[0]
    max_prob_index = np.argmax(probabilities)
    confidence = float(probabilities[max_prob_index])
    predicted_tag = label_encoder.inverse_transform([max_prob_index])[0]

    # Generate response
    if confidence < CONFIDENCE_THRESHOLD:
        bot_response = (
            "I'm not quite sure what you mean. "
            "Could you rephrase that? Type 'help' to see what I can do."
        )
    elif predicted_tag == "time":
        current_time = datetime.now().strftime("%I:%M %p")
        current_date = datetime.now().strftime("%A, %B %d, %Y")
        bot_response = f"It's currently {current_time} on {current_date}."
    else:
        bot_response = random.choice(responses.get(predicted_tag, ["I'm not sure how to respond."]))

    return jsonify({
        "response": bot_response,
        "intent": predicted_tag,
        "confidence": round(confidence, 3)
    })


# For local development
if __name__ == "__main__":
    app.run(debug=True, port=5000)
