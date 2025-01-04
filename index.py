from flask import Flask, render_template, request
import pickle
import numpy as np
import re

# Load the trained model and vectorizer
with open('fake_news_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)


def data_cleaning(text):
    # Lowercase the text
    text = text.lower()
    # Remove text inside square brackets
    text = re.sub('\[.*?\]', '', text)
    # Replace non-word characters with a space
    text = re.sub("\\W", " ", text) 
    # Remove URLs
    text = re.sub('https?://\S+|www\.\S+', '', text)
    # Remove HTML tags
    text = re.sub('<.*?>+', '', text)
    # Remove punctuation
    text = "".join([char for char in text if char not in string.punctuation])
    # Remove newlines
    text = re.sub('\n', '', text)
    # Remove words containing digits
    text = re.sub('\w*\d\w*', '', text)
    #Remove stop words
    text = " ".join([word for word in text.split() if word not in stopwords.words('english')])
    return text


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
        news_article=data_cleaning(news_article)
        
        # Clean and vectorize the article text
        article_tfidf = vectorizer.transform([news_article])

        # Predict if the article is fake or real
        prediction = model.predict(article_tfidf)

        # Return the result
        result = 'Fake News' if prediction == 0 else 'Real News'
        return render_template('index.html', prediction_result=result, news_article=news_article)

if __name__ == '__main__':
    app.run(debug=True)
