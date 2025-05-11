import pandas as pd
import re
import nltk
import joblib
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Tải stopwords nếu chưa có
nltk.download('stopwords', quiet=True)

# Đọc file spam.csv
print("Đang đọc dữ liệu...")
df = pd.read_csv("src/spam.csv", encoding="utf-8")[["Category", "Message"]]
df.columns = ['label', 'message']

# Thống kê dữ liệu
print(f"Tổng số mẫu: {len(df)}")
print(f"Số lượng email ham: {len(df[df['label'] == 'ham'])}")
print(f"Số lượng email spam: {len(df[df['label'] == 'spam'])}")

# Tiền xử lý văn bản
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = re.sub(r'\W', ' ', text.lower())
    words = [word for word in text.split() if word not in stop_words]
    return ' '.join(words)

print("Đang tiền xử lý dữ liệu...")
df['processed'] = df['message'].apply(preprocess)

# Chuyển đổi đặc trưng
print("Đang chuyển đổi đặc trưng...")
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['processed'])
y = df['label'].map({'ham': 0, 'spam': 1})

# Chia dữ liệu
print("Đang chia dữ liệu thành tập huấn luyện và kiểm tra...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"Số lượng mẫu huấn luyện: {X_train.shape[0]}")
print(f"Số lượng mẫu kiểm tra: {X_test.shape[0]}")

# Huấn luyện mô hình
print("Đang huấn luyện mô hình...")
model = MultinomialNB()
model.fit(X_train, y_train)

# Đánh giá mô hình
print("\n=== KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH ===")
y_pred = model.predict(X_test)

# Độ chính xác
accuracy = accuracy_score(y_test, y_pred)
print(f"\nĐộ chính xác (Accuracy): {accuracy:.4f}")

# Ma trận nhầm lẫn
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nMa trận nhầm lẫn (Confusion Matrix):")
print("                  Dự đoán: Ham | Dự đoán: Spam")
print(f"Thực tế: Ham      {conf_matrix[0][0]:10d} | {conf_matrix[0][1]:10d}")
print(f"Thực tế: Spam     {conf_matrix[1][0]:10d} | {conf_matrix[1][1]:10d}")

# Giải thích ma trận nhầm lẫn
tn, fp, fn, tp = conf_matrix.ravel()
print("\nGiải thích ma trận nhầm lẫn:")
print(f"True Negatives (TN): {tn} (email không phải spam được phân loại đúng)")
print(f"False Positives (FP): {fp} (email không phải spam bị phân loại nhầm thành spam)")
print(f"False Negatives (FN): {fn} (email spam bị phân loại nhầm thành không phải spam)")
print(f"True Positives (TP): {tp} (email spam được phân loại đúng)")

# Báo cáo phân loại chi tiết
print("\nBáo cáo phân loại chi tiết:")
report = classification_report(y_test, y_pred, target_names=['ham', 'spam'])
print(report)

# Tính toán các chỉ số khác
precision_ham = conf_matrix[0][0] / (conf_matrix[0][0] + conf_matrix[1][0])
precision_spam = conf_matrix[1][1] / (conf_matrix[0][1] + conf_matrix[1][1])
recall_ham = conf_matrix[0][0] / (conf_matrix[0][0] + conf_matrix[0][1])
recall_spam = conf_matrix[1][1] / (conf_matrix[1][0] + conf_matrix[1][1])
f1_ham = 2 * (precision_ham * recall_ham) / (precision_ham + recall_ham)
f1_spam = 2 * (precision_spam * recall_spam) / (precision_spam + recall_spam)

print("\nCác chỉ số đánh giá bổ sung:")
print(f"Precision (Ham): {precision_ham:.4f}")
print(f"Precision (Spam): {precision_spam:.4f}")
print(f"Recall (Ham): {recall_ham:.4f}")
print(f"Recall (Spam): {recall_spam:.4f}")
print(f"F1-score (Ham): {f1_ham:.4f}")
print(f"F1-score (Spam): {f1_spam:.4f}")

print("\nĐánh giá hoàn tất!")
