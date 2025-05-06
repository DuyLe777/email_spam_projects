import tkinter as tk
from tkinter import messagebox
import joblib
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = re.sub(r'\W', ' ', text.lower())
    words = [word for word in text.split() if word not in stop_words]
    return ' '.join(words)


model = joblib.load("spam_classifier_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def check_spam():
    text = entry.get("1.0", tk.END).strip()
    if not text:
        messagebox.showwarning("Cảnh báo", "Vui lòng nhập nội dung email.")
        return
    processed = preprocess(text)
    vect_text = vectorizer.transform([processed])
    prediction = model.predict(vect_text)
    result = "✅ Không phải thư rác" if prediction[0] == 0 else "⚠️ Đây là thư rác!"
    messagebox.showinfo("Kết quả", result)

root = tk.Tk()
root.title("Spam Classifier")

label = tk.Label(root, text="Nhập nội dung email:")
label.pack(pady=10)

entry = tk.Text(root, height=10, width=50)
entry.pack()

button = tk.Button(root, text="Kiểm tra", command=check_spam)
button.pack(pady=10)

root.mainloop()
