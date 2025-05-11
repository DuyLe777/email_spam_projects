from flask import Flask, render_template, request
import joblib # load mo hinh va vectorizer  luu san 
import re # thu vien reget xu li van ban
from nltk.corpus import stopwords # loai bo cac tu dừng trong tiếng anh

app = Flask(__name__)
model = joblib.load("spam_classifier_model.pkl") # mô hình đã được huấn luyện trước đó
vectorizer = joblib.load("vectorizer.pkl") # bộ biến đổi văn bản
stop_words = set(stopwords.words('english')) # Danh sách các từ dừng để loại bỏ khi xử lý dữ liệu đầu vào.

def preprocess(text):
    text = re.sub(r'\W', ' ', text.lower()) # chuyển văn bản thành chữ thường loại bỏ kí tự khong phải \w
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
