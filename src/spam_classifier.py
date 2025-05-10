import pandas as pd
import re
import nltk
import joblib
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Cài đặt NLTK
print("1. Cài đặt NLTK...")
nltk.download('stopwords')
print("✅ Đã cài đặt NLTK.")

# Xử lý dữ liệu
print("\n2. Xử lý dữ liệu...")
df = pd.read_csv("spam.csv", encoding="utf-8")[["Category", "Message"]]
df.columns = ['label', 'message']
print(f"✅ Đã tải dữ liệu: {len(df)} mẫu.")
print(f"- Thống kê nhãn:\n{df['label'].value_counts()}")

# Tiền xử lý văn bản
print("\n3. Tiền xử lý văn bản...")
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    # Chuyển thành chữ thường và loại bỏ ký tự đặc biệt
    text = re.sub(r'\W', ' ', text.lower())
    # Tách từ và loại bỏ stopwords
    words = [word for word in text.split() if word not in stop_words]
    # Áp dụng stemming
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

df['processed'] = df['message'].apply(preprocess)
print("✅ Đã tiền xử lý văn bản.")

# Vector hóa và chia dữ liệu
print("\n4. Vector hóa và chia dữ liệu...")
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['processed'])
y = df['label'].map({'ham': 0, 'spam': 1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print(f"✅ Đã vector hóa dữ liệu: {X.shape[0]} mẫu, {X.shape[1]} đặc trưng.")
print(f"- Tập huấn luyện: {X_train.shape[0]} mẫu")
print(f"- Tập kiểm thử: {X_test.shape[0]} mẫu")

# Huấn luyện mô hình và lưu
print("\n5. Huấn luyện mô hình...")
model = MultinomialNB()
model.fit(X_train, y_train)

# Lưu mô hình
print("\n6. Lưu mô hình...")
joblib.dump(model, 'spam_classifier_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print("✅ Đã huấn luyện và lưu mô hình.")

# Đánh giá mô hình
print("\n7. Đánh giá mô hình...")
y_pred = model.predict(X_test)

# Tính các chỉ số đánh giá
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Hiển thị kết quả
print("\n===== KẾT QUẢ ĐÁNH GIÁ =====")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

print("\nMa trận nhầm lẫn:")
print(f"[[TN={cm[0][0]} FP={cm[0][1]}]")
print(f" [FN={cm[1][0]} TP={cm[1][1]}]]")

print("\nBáo cáo phân loại chi tiết:")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

print("\n✅ Hoàn thành quá trình huấn luyện và đánh giá mô hình.")
