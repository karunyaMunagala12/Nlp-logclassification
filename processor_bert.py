import os
import joblib
import logging
from sentence_transformers import SentenceTransformer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Embedding Model
model_embedding = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight embedding model

# Check if the classifier model exists
model_path = "models/log_classifier.joblib"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model file '{model_path}' not found! Make sure it exists.")

# Load Model
model_classification = joblib.load(model_path)

def classify_with_bert(log_message):
    """
    Classifies a log message using a pre-trained BERT-based model.

    Parameters:
    - log_message (str): The log message to classify.

    Returns:
    - predicted_label (str): The predicted category or "Unclassified" if confidence is low.
    """
    try:
        embeddings = model_embedding.encode([log_message])
        probabilities = model_classification.predict_proba(embeddings)[0]

        if max(probabilities) < 0.5:
            return "Unclassified"
        
        predicted_label = model_classification.predict(embeddings)[0]
        logger.info(f"Log: {log_message} | Predicted: {predicted_label}")
        
        return predicted_label

    except Exception as e:
        logger.error(f"❌ Error in classify_with_bert: {str(e)}")
        return "Unclassified"

if __name__ == "__main__":
    logs = [
        "alpha.osapi_compute.wsgi.server - 12.10.11.1 - API returned 404 not found error",
        "GET /v2/3454/servers/detail HTTP/1.1 RCODE   404 len: 1583 time: 0.1878400",
        "System crashed due to drivers errors when restarting the server",
        "Hey bro, chill ya!",
        "Multiple login failures occurred on user 6454 account",
        "Server A790 was restarted unexpectedly during the process of data transfer"
    ]
    for log in logs:
        label = classify_with_bert(log)
        print(log, "->", label)