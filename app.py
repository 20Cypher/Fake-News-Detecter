from flask import Flask, request, jsonify
import pandas as pd
from fake_news_detection import manual_testing
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Initialize the TfidfVectorizer
vectorization = TfidfVectorizer()

CORS(app)
@app.route('/predict', methods=['POST'])
def predict():
    # Implement your machine learning model logic here
    data = request.get_json()
    text = data.get('text', '')
    # Make predictions and return results as JSON
    predictions = manual_testing(text)
    result = {"predictions": predictions}
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
