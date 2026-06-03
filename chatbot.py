"""
chatbot.py - AI Chatbot Application
=====================================

An NLP-powered chatbot that uses TF-IDF vectorization and Logistic Regression
to classify user intents and provide appropriate responses.

Usage:
    python chatbot.py

Make sure to run 'python train_model.py' first to generate the model files.
"""

import json
import os
import random
from datetime import datetime

import numpy as np
import joblib


# ============================================================================
# Configuration
# ============================================================================

MODEL_DIR = "models"
INTENTS_FILE = "intents.json"
CONFIDENCE_THRESHOLD = 0.25  # Minimum confidence to accept a prediction


# ============================================================================
# Chatbot Class
# ============================================================================

class ChatBot:
    """AI Chatbot using scikit-learn for intent classification.

    This chatbot processes user input through a TF-IDF vectorizer,
    classifies the intent using a trained Logistic Regression model,
    and returns an appropriate response from predefined templates.

    Attributes:
        classifier: Trained Logistic Regression model.
        vectorizer: Fitted TF-IDF vectorizer.
        label_encoder: Fitted label encoder for intent tags.
        intents_data: Dictionary of intents with responses.
        confidence_threshold: Minimum prediction confidence to accept.
    """

    def __init__(self, model_dir: str = MODEL_DIR, intents_file: str = INTENTS_FILE):
        """Initialize the chatbot by loading model and intents data.

        Args:
            model_dir: Directory containing trained model files.
            intents_file: Path to the intents JSON file.
        """
        self.confidence_threshold = CONFIDENCE_THRESHOLD
        self._load_model(model_dir)
        self._load_intents(intents_file)

    def _load_model(self, model_dir: str):
        """Load trained model components from disk.

        Args:
            model_dir: Directory containing .pkl model files.

        Raises:
            FileNotFoundError: If model files are not found.
        """
        classifier_path = os.path.join(model_dir, "intent_classifier.pkl")
        vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
        encoder_path = os.path.join(model_dir, "label_encoder.pkl")

        # Check if model files exist
        for path in [classifier_path, vectorizer_path, encoder_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Model file not found: {path}\n"
                    "Please run 'python train_model.py' first to train the model."
                )

        self.classifier = joblib.load(classifier_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.label_encoder = joblib.load(encoder_path)

    def _load_intents(self, intents_file: str):
        """Load intents data for response retrieval.

        Args:
            intents_file: Path to the intents JSON file.
        """
        with open(intents_file, "r", encoding="utf-8") as f:
            self.intents_data = json.load(f)

        # Build a quick lookup: tag -> list of responses
        self.responses = {}
        for intent in self.intents_data["intents"]:
            self.responses[intent["tag"]] = intent["responses"]

    def predict_intent(self, user_input: str) -> tuple:
        """Predict the intent of user input.

        Transforms the input text using the TF-IDF vectorizer and
        predicts the intent class using the trained classifier.

        Args:
            user_input: The user's message text.

        Returns:
            Tuple of (predicted_tag, confidence_score).
        """
        # Preprocess: lowercase the input
        processed_input = user_input.lower().strip()

        # Vectorize the input using the trained TF-IDF vectorizer
        input_vector = self.vectorizer.transform([processed_input])

        # Get prediction probabilities for all classes
        probabilities = self.classifier.predict_proba(input_vector)[0]

        # Get the highest confidence prediction
        max_prob_index = np.argmax(probabilities)
        confidence = probabilities[max_prob_index]

        # Decode the predicted label back to intent tag
        predicted_tag = self.label_encoder.inverse_transform([max_prob_index])[0]

        return predicted_tag, confidence

    def get_response(self, tag: str) -> str:
        """Get a response for the predicted intent tag.

        Handles special cases like the 'time' intent which requires
        dynamic content generation.

        Args:
            tag: The predicted intent tag.

        Returns:
            A response string appropriate for the intent.
        """
        # Special case: time intent returns the current time
        if tag == "time":
            current_time = datetime.now().strftime("%I:%M %p")
            current_date = datetime.now().strftime("%A, %B %d, %Y")
            return f"It's currently {current_time} on {current_date}."

        # Get responses for the tag and pick one randomly
        responses = self.responses.get(tag, ["I'm not sure how to respond to that."])
        return random.choice(responses)

    def chat(self, user_input: str) -> str:
        """Process user input and return a chatbot response.

        This is the main method that ties together intent prediction
        and response generation.

        Args:
            user_input: The user's message text.

        Returns:
            The chatbot's response string.
        """
        # Handle empty input
        if not user_input.strip():
            return "I didn't catch that. Could you say something?"

        # Predict intent and confidence
        tag, confidence = self.predict_intent(user_input)

        # If confidence is too low, return a fallback response
        if confidence < self.confidence_threshold:
            return (
                "I'm not quite sure what you mean. "
                "Could you rephrase that? Type 'help' to see what I can do."
            )

        # Get and return the response
        return self.get_response(tag)


# ============================================================================
# Command-Line Interface
# ============================================================================

def print_banner():
    """Display the chatbot welcome banner."""
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║              🤖 AI ChatBot v1.0                         ║")
    print("║    Powered by scikit-learn & NLP                        ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║  Type your message and press Enter to chat.             ║")
    print("║  Type 'quit' or 'exit' to end the conversation.         ║")
    print("║  Type 'help' to see what I can do.                      ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()


def main():
    """Run the chatbot in interactive command-line mode."""
    print_banner()

    # Initialize the chatbot
    try:
        bot = ChatBot()
        print("[System] ChatBot loaded successfully!\n")
    except FileNotFoundError as e:
        print(f"[Error] {e}")
        print("[System] Please run 'python train_model.py' first.")
        return

    # Main conversation loop
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()

            # Check for exit commands
            if user_input.lower() in ("quit", "exit", "q"):
                print("\nChatBot: Goodbye! Have a wonderful day! 👋\n")
                break

            # Skip empty input
            if not user_input:
                continue

            # Get chatbot response
            response = bot.chat(user_input)
            print(f"ChatBot: {response}\n")

        except KeyboardInterrupt:
            print("\n\nChatBot: Goodbye! See you next time! 👋\n")
            break
        except Exception as e:
            print(f"[Error] Something went wrong: {e}\n")


if __name__ == "__main__":
    main()
