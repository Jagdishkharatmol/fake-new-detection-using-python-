from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the trained model and vectorizer
with open('fake_news_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

# Initialize Flask app
app = Flask(__name__)

# Route to display the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle the form submission and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news_article = request.form['news_article']  # Get the article text from the form

        # Clean and vectorize the article text
        article_tfidf = vectorizer.transform([news_article])

        # Predict if the article is fake or real
        prediction = model.predict(article_tfidf)

        # Return the result
        result = 'Fake News' if prediction == 0 else 'Real News'
        return render_template('index.html', prediction_result=result, news_article=news_article)

if __name__ == '__main__':
    app.run(debug=True)
