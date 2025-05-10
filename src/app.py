from flask import Flask, render_template, request
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Đảm bảo stopwords đã được tải
nltk.download('stopwords')

app = Flask(__name__)
model = joblib.load("spam_classifier_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    # Chuyển thành chữ thường và loại bỏ ký tự đặc biệt
    text = re.sub(r'\W', ' ', text.lower())
    # Tách từ và loại bỏ stopwords
    words = [word for word in text.split() if word not in stop_words]
    # Áp dụng stemming
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    probability = None
    
    if request.method == "POST":
        email = request.form["email"]
        processed = preprocess(email)
        vect = vectorizer.transform([processed])
        prediction = model.predict(vect)[0]
        prob = model.predict_proba(vect)[0][1]  # Lấy xác suất là spam
        
        result = "SPAM" if prediction == 1 else "Không phải SPAM"
        probability = f"{prob:.2%}"
        
    return render_template("index.html", result=result, probability=probability, email=request.form.get("email", ""))

if __name__ == "__main__":
    app.run(debug=True)
