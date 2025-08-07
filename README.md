# 🧠 backEnd_nlpSentimentApp

This is the backend API for the **NLP Sentiment Analysis Web App**, built using **Python** and connected to the **OpenAI language model**. It classifies Amazon product reviews as either **"Positive"** or **"Negative"** using real-time inference.

This backend serves as the core of the intelligent system and integrates with a **React.js frontend** to provide a seamless user experience for sentiment classification.

---

## 🚀 Key Features

- ✅ **RESTful API** with a `/predict` endpoint (Flask)
- ✅ **Text preprocessing pipeline** based on notebook logic
- ✅ **Integration with OpenAI API** for real-time sentiment prediction
- ✅ **Input validation and error handling**
- ✅ **Basic test suite** for backend and model inference
- ✅ **Notebook** for training, cleaning, and evaluation
- ✅ **Ready for deployment** (cloud hosting)

---

## 🧹 Preprocessing Steps

The API uses a preprocessing pipeline to clean and standardize user input before inference:

- Convert to lowercase
- Remove punctuation and stopwords
- (Optional) Tokenization or chunking
- (Optional) Padding or truncation
- Filtering short or irrelevant reviews

---

## 📦 Project Structure
├── app.py # API entry point

├── openai_client.py # Wrapper for OpenAI API requests

├── preprocessing.py # Text cleaning and processing functions

├── notebook_model.ipynb # Training and data exploration

├── test_backend.py # Basic unit tests

├── requirements.txt # Dependencies

└── README.md # Project documentation


## EXAMPLE OF JSON RESPONSE:

{
  "sentiment": "Positive",
  "confidence": 0.93
}


## 🧪 Development & Testing

### To run locally:

#### run in console to Install dependencies:

pip install -r requirements.txt

#### run in console to Start the server
python app.py

## To test the API:

### run in console

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I hated this product."}'


