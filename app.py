from flask import Flask, render_template, request
import pickle
import sklearn

# Initialize Flask app
app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open('fake_news_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news_text = request.form['news']

    # Transform input using the loaded TF-IDF vectorizer
    transformed_text = vectorizer.transform([news_text])

    # Predict using the loaded model
    prediction = model.predict(transformed_text)[0]

    result = "🟢 REAL News" if prediction == 0 else "🔴 FAKE News"

    return render_template('index.html', prediction=result, input_text=news_text)

if __name__ == '__main__':
    app.run(debug=True)
