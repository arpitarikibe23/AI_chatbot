# 🤖 AI ChatBot — NLP Intent Classifier

A smart conversational chatbot built with Python and scikit-learn that uses Natural Language Processing (NLP) to understand user intent and provide relevant responses.

## 📖 About

This project demonstrates core AI/ML concepts by implementing an intent-based chatbot from scratch. The bot classifies user messages into predefined categories (intents) and responds appropriately — no cloud APIs required.

**Key Concepts Demonstrated:**
- Text preprocessing and vectorization (TF-IDF)
- Multi-class text classification (Logistic Regression)
- Model training, evaluation, and persistence
- Clean software architecture with separation of concerns

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| **Python 3.9+** | Core programming language |
| **NumPy** | Numerical computations and array operations |
| **pandas** | Data manipulation and training data preparation |
| **scikit-learn** | TF-IDF vectorization, Logistic Regression, model evaluation |
| **joblib** | Model serialization and persistence |

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/ai-chatbot.git
   cd ai-chatbot
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux/Mac
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. **Train the model:**
   ```bash
   python train_model.py
   ```
   This processes the intent data, trains the classifier, and saves model files to the `models/` directory.

2. **Start the chatbot:**
   ```bash
   python chatbot.py
   ```

3. **Chat away!** Type your messages and press Enter. Type `quit` to exit.

### Example Conversation

```
You: Hello!
ChatBot: Hi there! What can I do for you?

You: Tell me a joke
ChatBot: Why do programmers prefer dark mode? Because light attracts bugs!

You: What time is it?
ChatBot: It's currently 02:30 PM on Monday, January 15, 2024.

You: Thanks!
ChatBot: Happy to help!

You: quit
ChatBot: Goodbye! Have a wonderful day! 👋
```

## 🧠 How It Works

### Architecture Overview

```
User Input → Preprocessing → TF-IDF Vectorization → Classification → Response Selection
```

### Step-by-Step:

1. **Training Phase (`train_model.py`):**
   - Loads intent patterns from `intents.json`
   - Converts text patterns into a pandas DataFrame
   - Applies TF-IDF (Term Frequency-Inverse Document Frequency) vectorization to transform text into numerical feature vectors
   - Trains a Logistic Regression classifier on the TF-IDF features
   - Saves the trained model, vectorizer, and label encoder to disk

2. **Inference Phase (`chatbot.py`):**
   - Loads the pre-trained model components
   - Takes user input and preprocesses it (lowercase, strip whitespace)
   - Transforms input through the same TF-IDF vectorizer
   - Predicts the intent class with confidence score
   - If confidence exceeds threshold → returns a matching response
   - If confidence is too low → returns a fallback message

### What is TF-IDF?

**TF-IDF (Term Frequency-Inverse Document Frequency)** is a statistical measure that evaluates how relevant a word is to a document within a collection. It works by:

- **TF (Term Frequency):** How often a word appears in a single document
- **IDF (Inverse Document Frequency):** How rare the word is across all documents

Words that appear frequently in one document but rarely across others get higher scores, making them more useful for classification.

## 📁 Project Structure

```
ai-chatbot/
├── chatbot.py          # Main chatbot application (inference)
├── train_model.py      # Model training script
├── intents.json        # Training data (patterns + responses)
├── requirements.txt    # Python dependencies
├── .gitignore          # Git ignore rules
├── README.md           # This file
└── models/             # Generated after training (git-ignored)
    ├── intent_classifier.pkl
    ├── tfidf_vectorizer.pkl
    └── label_encoder.pkl
```

## 🔧 Customization

### Adding New Intents

Edit `intents.json` to add new intents:

```json
{
  "tag": "your_new_intent",
  "patterns": [
    "example pattern 1",
    "example pattern 2",
    "example pattern 3"
  ],
  "responses": [
    "Response option 1",
    "Response option 2"
  ]
}
```

Then retrain the model:
```bash
python train_model.py
```

## 🚀 Potential Extensions

This project serves as a foundation that can be extended with:

- **OpenAI API Integration:** Replace or augment the classifier with GPT-4 for more natural, context-aware conversations
- **LangChain Framework:** Build complex conversational chains with memory and tool usage
- **RAG (Retrieval-Augmented Generation):** Connect to a knowledge base for factual, document-grounded responses
- **TensorFlow/PyTorch:** Replace Logistic Regression with a deep learning model (LSTM, Transformer) for improved accuracy
- **Sentiment Analysis:** Add emotional awareness to responses
- **Multi-turn Context:** Implement conversation history for contextual follow-ups
- **Voice Interface:** Add speech-to-text and text-to-speech capabilities
- **Web UI:** Build a Flask/FastAPI frontend for browser-based interaction

## 📊 Model Performance

The model achieves high cross-validation accuracy on the training data due to the clear separation between intent categories. For production use, consider:

- Adding more diverse training patterns per intent
- Implementing out-of-scope detection
- Using more sophisticated models for ambiguous inputs

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

*Built with ❤️ and Python. A demonstration of practical NLP and machine learning.*
