
# Import necessary modules for machine learning
from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm

app = Flask(__name__, template_folder="template")

# Load the model and CountVectorizer
spam_model = svm.SVC()
cv = CountVectorizer()

# Load the CSV data into a Pandas DataFrame
spam = pd.read_csv('data/spam.csv')

# Feature extraction
features = cv.fit_transform(spam['message'])

# Train the model
spam_model.fit(features, spam['label'])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = [request.form['message']]
        message_vectorized = cv.transform(message)
        prediction = spam_model.predict(message_vectorized)
        return render_template('index.html', message=message[0], prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
