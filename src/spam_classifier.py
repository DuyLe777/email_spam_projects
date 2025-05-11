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

nltk.download('stopwords') # tải danh sách từ dừng trong tiếng anh 


#Đọc file spam.csv, chỉ lấy 2 cột: Category (ham/spam), Message (nội dung email).
df = pd.read_csv("spam.csv", encoding="utf-8")[["Category", "Message"]]
df.columns = ['label', 'message'] # Đổi tên cột cho dễ dùng: label và message.


# Tiền xử lí văn bản
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))


def preprocess(text):
    text = re.sub(r'\W', ' ', text.lower())
    words = [word for word in text.split() if word not in stop_words]
    return ' '.join(words)

# Tạo cột mới processed chứa văn bản đã được làm sạch
df['processed'] = df['message'].apply(preprocess)

vectorizer = CountVectorizer() # chuyển văn bản sang ma trận tần xuất
X = vectorizer.fit_transform(df['processed'])
y = df['label'].map({'ham': 0, 'spam': 1}) # y: Chuyển label từ "ham"/"spam" sang 0/1

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

# Dự đoán trên tập test
y_pred = model.predict(X_test)

# In kết quả huấn luyện
print("\n=== KẾT QUẢ HUẤN LUYỆN MÔ HÌNH ===")
print(f"\nSố lượng mẫu huấn luyện: {X_train.shape[0]}")
print(f"Số lượng mẫu kiểm tra: {X_test.shape[0]}")

# In độ chính xác
accuracy = accuracy_score(y_test, y_pred)
print(f"\nĐộ chính xác (Accuracy): {accuracy:.4f}")

# In ma trận nhầm lẫn
print("\nMa trận nhầm lẫn:")
print(confusion_matrix(y_test, y_pred))

# In báo cáo phân loại chi tiết
print("\nBáo cáo phân loại chi tiết:")
print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))

# Lưu mô hình và vectorizer
joblib.dump(model, 'spam_classifier_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("\nĐã huấn luyện và lưu mô hình.")
