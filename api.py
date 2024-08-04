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

with open('Models/model_xgb.pkl', 'rb') as f:
    xgboost_model = pickle.load(f)

with open('Models/countVectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('Models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load the Hugging Face sentiment analysis pipeline for general sentiment
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
    "Joy", "Excited", "Optimistic", "Content", "Happy", "Pleased", "Delighted", "Satisfied",
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
    # General sentiment prediction using Hugging Face model
    hf_result = hf_sentiment_pipeline(text_input)[0]
    general_sentiment = hf_result['label']
    general_score = hf_result['score']
    
    # Use Twitter RoBERTa model to get a more specific sentiment score
    twitter_result = twitter_sentiment_pipeline(text_input)[0]
    
    # Adjust the specific score based on the general sentiment
    if general_sentiment == 'POSITIVE':
        specific_score = twitter_result['score']
    elif general_sentiment == 'NEGATIVE':
        specific_score = 1 - twitter_result['score']  # Invert the score for negative sentiment
    else:
        specific_score = 0.5  # Neutral case
    
    # Determine specific sentiment based on general sentiment and score
    if general_sentiment == 'POSITIVE':
        if specific_score >= 0.8:
            specific_sentiment = random.choice(POSITIVE_SENTIMENTS[:5])  # Stronger positive emotions
        elif 0.6 <= specific_score < 0.8:
            specific_sentiment = random.choice(POSITIVE_SENTIMENTS[5:10])  # Moderate positive emotions
        else:
            specific_sentiment = random.choice(POSITIVE_SENTIMENTS[10:])  # Milder positive emotions
    elif general_sentiment == 'NEGATIVE':
        if specific_score >= 0.8:
            specific_sentiment = random.choice(NEGATIVE_SENTIMENTS[:5])  # Stronger negative emotions
        elif 0.6 <= specific_score < 0.8:
            specific_sentiment = random.choice(NEGATIVE_SENTIMENTS[5:10])  # Moderate negative emotions
        else:
            specific_sentiment = random.choice(NEGATIVE_SENTIMENTS[10:])  # Milder negative emotions
    else:
        specific_sentiment = random.choice(NEUTRAL_SENTIMENTS)
    
    # Subjectivity analysis using TextBlob
    blob = TextBlob(text_input)
    subjectivity = blob.sentiment.subjectivity
    
    return general_sentiment, specific_sentiment, specific_score, subjectivity

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

        # Subjectivity analysis using TextBlob
        blob = TextBlob(chunk.iloc[i][column_to_analyze])
        subjectivity = blob.sentiment.subjectivity

        corpus.append(sentiment)
        subjectivities.append(subjectivity)

    chunk["Predicted sentiment"] = corpus
    chunk["Subjectivity"] = subjectivities
    return chunk

def find_text_column(data):
    for column in data.columns:
        # Check if the column seems to contain text data
        sample_texts = data[column].dropna().astype(str).sample(n=5, random_state=0)
        if all(sample_texts.str.len().between(10, 1000)):  # Simple check for non-empty, reasonably long text
            return column
    return None

def bulk_prediction(data, text_column, chunk_size=300):
    # Process data in chunks
    chunks = pd.read_csv(BytesIO(data.to_csv(index=False)), chunksize=chunk_size)
    processed_chunks = []

    for chunk in tqdm(chunks, total=(len(data) // chunk_size) + 1):
        # Apply sentiment analysis to each chunk
        chunk[['Sentiment Predictions', 'Sentiment Score', 'Subjectivity']] = chunk[text_column].apply(lambda x: analyze_sentiment(x), result_type='expand')
        processed_chunks.append(chunk)

    # Concatenate all processed chunks into a single DataFrame
    result_data = pd.concat(processed_chunks, ignore_index=True)

    # Save predictions to CSV
    predictions_csv = BytesIO()
    result_data.to_csv(predictions_csv, index=False)
    predictions_csv.seek(0)

    # Generate sentiment distribution graph
    graph = get_distribution_graph(result_data)

    return predictions_csv, graph

def analyze_sentiment(text):
    result = hf_sentiment_pipeline(text)[0]
    sentiment = result['label']
    score = result['score']
    
    # Subjectivity analysis using TextBlob
    blob = TextBlob(text)
    subjectivity = blob.sentiment.subjectivity
    
    return pd.Series([sentiment, score, subjectivity], index=['Sentiment Predictions', 'Sentiment Score', 'Subjectivity'])

def get_distribution_graph(data):
    fig, ax = plt.subplots(figsize=(5, 5))
    colors = ("green", "red", "blue")
    wp = {"linewidth": 1, "edgecolor": "black"}
    tags = data["Sentiment Predictions"].value_counts()
    explode = (0.01, 0.01, 0.01)

    tags.plot(
        kind="pie",
        autopct="%1.1f%%",
        shadow=True,
        colors=colors,
        startangle=90,
        wedgeprops=wp,
        explode=explode,
        title="Sentiment Distribution",
        xlabel="",
        ylabel="",
        ax=ax
    )

    graph = BytesIO()
    plt.savefig(graph, format="png")
    plt.close(fig)

    return graph

if __name__ == "__main__":
    app.run(port=5000, debug=True)
