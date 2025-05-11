# Spam Classifier Project


## 📝 Mô tả

Dự án này là một hệ thống phân loại thư rác sử dụng thuật toán Multinomial Naive Bayes. Hệ thống có thể phân biệt giữa thư thông thường (ham) và thư rác (spam) dựa trên nội dung của thư. Dự án được phát triển nhằm mục đích học tập và nghiên cứu về ứng dụng của Machine Learning trong xử lý ngôn ngữ tự nhiên.

## 🚀 Tính năng chính

- **Xử lý văn bản tự động**:
  - Loại bỏ ký tự đặc biệt
  - Chuyển đổi chữ thường
  - Loại bỏ stopwords
  - Stemming từ

- **Phân loại thư rác**:
  - Sử dụng Multinomial Naive Bayes
  - Hiển thị xác suất dự đoán
  - Độ chính xác cao

- **Giao diện web**:
  - Form nhập liệu đơn giản
  - Hiển thị kết quả trực quan
  - Responsive design

## 🛠️ Công nghệ sử dụng

- **Ngôn ngữ lập trình**: Python 3.x
- **Machine Learning**: scikit-learn
- **NLP**: NLTK (Natural Language Toolkit)
- **Web Framework**: Flask
- **Data Processing**: pandas
- **Model Serialization**: joblib

## 📋 Yêu cầu hệ thống

- Python 3.x
- pip (Python package manager)
- Các thư viện được liệt kê trong `requirements.txt`

## 🚀 Cài đặt và Chạy

1. Clone repository:
```bash
git clone https://github.com/DuyLe777/spam-classifier-project.git
cd spam-classifier-project
```

2. Tạo môi trường ảo:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Cài đặt dependencies:
```bash
pip install -r requirements.txt
```

4. Chạy ứng dụng:
```bash
python src/app.py
```

5. Truy cập ứng dụng tại: `http://localhost:5000`

## 📊 Cấu trúc dự án

```
spam-classifier-project/
├── src/
│   ├── spam_classifier.py    # Mô hình ML và xử lý dữ liệu
│   ├── app.py               # Web application
│   ├── spam.csv             # Dataset gốc
│   ├── spam_classifier_model.pkl    # Model đã train
│   ├── vectorizer.pkl       # Vectorizer đã train
│   ├── static/              # CSS, JS files
│   └── templates/           # HTML templates
├── requirements.txt         # Dependencies
└── README.md               # Documentation
```


