from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import re
from io import BytesIO
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
import pandas as pd
import base64
import logging
from logging.handlers import RotatingFileHandler
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm import tqdm
import random
import pickle
from collections import Counter
from textblob import TextBlob

STOPWORDS = set(stopwords.words("english"))

app = Flask(__name__)
CORS(app)

# Set up logging
handler = RotatingFileHandler('flask_app.log', maxBytes=10000, backupCount=1)
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)

# Load models for general sentiment
with open('Models/model_xgb.pkl', 'rb') as f:
    xgboost_model = pickle.load(f)

with open('Models/countVectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('Models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

hf_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
hf_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
hf_sentiment_pipeline = pipeline("sentiment-analysis", model=hf_model, tokenizer=hf_tokenizer)

# Load the specific sentiment model (Twitter RoBERTa)
twitter_sentiment_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
twitter_sentiment_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
twitter_sentiment_pipeline = pipeline("sentiment-analysis", model=twitter_sentiment_model, tokenizer=twitter_sentiment_tokenizer)

# Define the specific sentiment mapping
specific_sentiment_mapping = {
    "Positive": ["joy", "surprise", "optimistic", "content", "trust"],
    "Negative": ["anger", "disgust", "fear", "sadness"],
    "Neutral": ["neutral", "curiosity"]
}

# Define keywords for aspect-based sentiment analysis
aspect_keywords = {
    'Delivery': ['delivery', 'shipping', 'shipment', 'delivered'],
    'Product Quality': ['quality', 'product', 'item', 'goods'],
    'Usage': ['students', 'elderly', 'children']
}

@app.route("/test", methods=["GET"])
def test():
    return "Test request received successfully. Service is running."

@app.route("/test_post", methods=["POST"])
def test_post():
    return "POST request received successfully"

@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("landing.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" in request.files:
            # Bulk prediction from CSV file
            file = request.files["file"]
            data = pd.read_csv(file)

            text_column = find_text_column(data)

            if text_column is None:
                return jsonify({"error": "No suitable text column found in the CSV"}), 400

            # Perform bulk prediction
            predictions_csv, graph = bulk_prediction(data, text_column)

            response = send_file(
                predictions_csv,
                mimetype="text/csv",
                as_attachment=True,
                download_name="Predictions.csv",
            )

            response.headers["X-Graph-Exists"] = "true"
            response.headers["X-Graph-Data"] = base64.b64encode(
                graph.getbuffer()
            ).decode("ascii")

            return response

        elif "text" in request.json:
            # Single string prediction
            text_input = request.json["text"]
            aspect = identify_aspect(text_input)
            general_sentiment, specific_sentiment, specific_score, subjectivity = single_prediction(text_input)

            return jsonify({
                "aspect": aspect,
                "general_sentiment": general_sentiment,
                "specific_sentiment": specific_sentiment,
                "specific_score": specific_score,
                "subjectivity": subjectivity
            })

    except Exception as e:
        return jsonify({"error": str(e)})

POSITIVE_SENTIMENTS = [
    "Excited", "Optimistic", "Content", "Happy", "Pleased", "Delighted", "Satisfied",
    "Cheerful", "Grateful"
]

NEGATIVE_SENTIMENTS = [
    "Sad", "Angry", "Frustrated", "Disappointed", "Worried", "Annoyed", "Displeased", "Upset"
]

NEUTRAL_SENTIMENTS = [
    "Neutral", "Indifferent", "Ambivalent", "Impartial", "Unbiased", "Detached", "Uninvolved",
    "Dispassionate", "Noncommittal", "Objective"
]

def identify_aspect(text):
    aspect = "General"
    for key, keywords in aspect_keywords.items():
        for keyword in keywords:
            if keyword.lower() in text.lower():
                aspect = key
                break
    return aspect

def single_prediction(text_input):
    # Preprocess the input text for XGBoost
    processed_text = preprocess_text(text_input)
    xgboost_features = vectorizer.transform([processed_text])
    xgboost_features = scaler.transform(xgboost_features.toarray())

    # Predict sentiment using XGBoost
    xgboost_prediction = xgboost_model.predict(xgboost_features)[0]
    xgboost_proba = xgboost_model.predict_proba(xgboost_features).max()

    # Predict sentiment using Hugging Face model
    hf_result = hf_sentiment_pipeline(text_input)[0]
    hf_sentiment = hf_result['label']
    hf_score = hf_result['score']

    # Map XGBoost prediction to corresponding sentiment label
    if xgboost_prediction == 1:
        xgboost_sentiment = 'POSITIVE'
    elif xgboost_prediction == 0:
        xgboost_sentiment = 'NEGATIVE'
    else:
        xgboost_sentiment = 'NEUTRAL'

    # Combine predictions (example: weighted average of scores)
    combined_score = (hf_score + xgboost_proba) / 2
    if hf_sentiment == xgboost_sentiment:
        general_sentiment = hf_sentiment
    else:
        # Use weighted logic or prioritize one model if they differ
        general_sentiment = hf_sentiment if hf_score >= xgboost_proba else xgboost_sentiment

    # Perform specific sentiment mapping based on the combined sentiment
    specific_sentiment, specific_score = map_specific_sentiment(general_sentiment, combined_score)

    # Subjectivity analysis using TextBlob
    blob = TextBlob(text_input)
    subjectivity = blob.sentiment.subjectivity

    return general_sentiment, specific_sentiment, specific_score, subjectivity

def preprocess_text(text):
    # Preprocess text for XGBoost model (e.g., stemming, removing stopwords)
    review = re.sub("[^a-zA-Z]", " ", text)
    review = review.lower().split()
    review = [PorterStemmer().stem(word) for word in review if word not in STOPWORDS]
    return " ".join(review)

def map_specific_sentiment(general_sentiment, score):
    # Determine specific sentiment based on general sentiment and score
    if general_sentiment == 'POSITIVE':
        if score >= 0.8:
            specific_sentiment = random.choice(POSITIVE_SENTIMENTS[:5])
        elif 0.6 <= score < 0.8:
            specific_sentiment = random.choice(POSITIVE_SENTIMENTS[5:10])
        else:
            specific_sentiment = random.choice(POSITIVE_SENTIMENTS[10:])
    elif general_sentiment == 'NEGATIVE':
        if score >= 0.8:
            specific_sentiment = random.choice(NEGATIVE_SENTIMENTS[:5])
        elif 0.6 <= score < 0.8:
            specific_sentiment = random.choice(NEGATIVE_SENTIMENTS[5:10])
        else:
            specific_sentiment = random.choice(NEGATIVE_SENTIMENTS[10:])
    else:
        specific_sentiment = random.choice(NEUTRAL_SENTIMENTS)
    
    return specific_sentiment, score

senti_analyzer = SentimentIntensityAnalyzer()

def perform_sentiment_analysis(chunk, column_to_analyze):
    stemmer = PorterStemmer()
    corpus = []
    subjectivities = []

    for i in range(0, chunk.shape[0]):
        review = str(chunk.iloc[i][column_to_analyze])
        if column_to_analyze == 'ratings':
            review = " ".join(TextBlob(review).words)
        else:
            review = re.sub("[^a-zA-Z]", " ", review)
        review = review.lower().split()
        review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
        review = " ".join(review)

        # Perform sentiment analysis using NLTK's VADER
        sentiment_scores = senti_analyzer.polarity_scores(review)
        # Determine sentiment based on compound score
        if sentiment_scores['compound'] >= 0.05:
            sentiment = 'Positive'
        elif sentiment_scores['compound'] <= -0.05:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'

        corpus.append(sentiment)
        subjectivities.append(TextBlob(review).sentiment.subjectivity)

    return corpus, subjectivities

def bulk_prediction(data, column_to_analyze):
    sentiments, subjectivities = perform_sentiment_analysis(data, column_to_analyze)
    data['Sentiment'] = sentiments
    data['Subjectivity'] = subjectivities

    sentiment_distribution = Counter(data['Sentiment'])
    labels, values = zip(*sentiment_distribution.items())

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_title("Sentiment Distribution")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")

    img = BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)

    output_csv = BytesIO()
    data.to_csv(output_csv, index=False)
    output_csv.seek(0)

    return output_csv, img

def find_text_column(data):
    # Identify the text column in the CSV
    text_column = None
    for column in data.columns:
        if data[column].dtype == object:
            if data[column].str.contains(r'\w+', na=False).all():
                text_column = column
                break
    return text_column

if __name__ == "__main__":
    app.run(debug=True)
