# Email Spam Classifier

Ứng dụng phân loại email thành spam hoặc không phải spam (ham) sử dụng kỹ thuật học máy.

## Mô tả dự án

Dự án này xây dựng một hệ thống phân loại email tự động, giúp người dùng nhận diện thư rác (spam) và thư thông thường (ham). Hệ thống được triển khai dưới hai hình thức:
- Ứng dụng desktop với giao diện đồ họa (GUI) sử dụng Tkinter
- Ứng dụng web với giao diện hiện đại sử dụng Flask

## Cấu trúc dự án

```
email_spam_projects/
│
├── src/
│   ├── app.py                    # Ứng dụng web Flask
│   ├── spam_classifier.py        # Script huấn luyện mô hình
│   ├── spam_gui.py               # Ứng dụng desktop Tkinter
│   ├── evaluate_model.py         # Script đánh giá mô hình
│   ├── spam.csv                  # Dữ liệu huấn luyện
│   ├── spam_classifier_model.pkl # Mô hình đã huấn luyện
│   ├── vectorizer.pkl            # Bộ chuyển đổi văn bản thành vector
│   ├── static/                   # Thư mục chứa CSS, JS cho web app
│   └── templates/                # Thư mục chứa HTML cho web app
│
└── requirements.txt              # Danh sách các thư viện cần thiết
```

## Yêu cầu hệ thống

- Python 3.7 trở lên
- Các thư viện Python được liệt kê trong file requirements.txt

## Cài đặt và thiết lập

### 1. Clone dự án (nếu sử dụng Git)

```bash
git clone <repository-url>
cd email_spam_projects
```

### 2. Tạo và kích hoạt môi trường ảo (khuyến nghị)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Cài đặt các thư viện cần thiết

```bash
pip install -r requirements.txt
```

## Hướng dẫn sử dụng

### 1. Huấn luyện mô hình (nếu cần)

Mô hình đã được huấn luyện sẵn và lưu trong các file `.pkl`. Nếu bạn muốn huấn luyện lại mô hình:

```bash
python src/spam_classifier.py
```

### 2. Đánh giá mô hình

Để xem các chỉ số đánh giá chi tiết của mô hình:

```bash
python src/evaluate_model.py
```



### 3. Chạy ứng dụng web    

```bash
cd src
python app.py
```

Sau khi chạy, truy cập ứng dụng web tại địa chỉ: http://localhost:5000

## Ứng dụng web



1. Truy cập địa chỉ http://localhost:5000
2. Nhập nội dung email vào ô văn bản
3. Nhấn nút "Kiểm tra ngay"
4. Kết quả sẽ hiển thị trên trang web

## Công nghệ sử dụng

- **Ngôn ngữ lập trình**: Python
- **Thư viện xử lý dữ liệu**: Pandas, NLTK, Regex
- **Thư viện học máy**: Scikit-learn, Joblib
- **Framework web**: Flask
- **Giao diện web**: HTML, CSS, Bootstrap, Font Awesome

## Kết quả đánh giá mô hình

- **Độ chính xác tổng thể**: 97,85%
- **Precision (Ham)**: 99,06%
- **Precision (Spam)**: 90,32%
- **Recall (Ham)**: 98,45%
- **Recall (Spam)**: 93,96%
- **F1-score (Ham)**: 98,75%
- **F1-score (Spam)**: 92,11%

