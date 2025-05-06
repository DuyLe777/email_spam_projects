from flask import Flask, render_template, request
import joblib
import re
from nltk.corpus import stopwords

app = Flask(__name__)
model = joblib.load("spam_classifier_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = re.sub(r'\W', ' ', text.lower())
    words = [word for word in text.split() if word not in stop_words]
    return ' '.join(words)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        email = request.form["email"]
        processed = preprocess(email)
        vect = vectorizer.transform([processed])
        prediction = model.predict(vect)[0]
        result = "SPAM" if prediction == 1 else "Không phải thư rác"
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
