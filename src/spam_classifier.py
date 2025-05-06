import pandas as pd
import re
import nltk
import joblib
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

nltk.download('stopwords')

df = pd.read_csv("spam.csv", encoding="utf-8")[["Category", "Message"]]
df.columns = ['label', 'message']

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))


def preprocess(text):
    text = re.sub(r'\W', ' ', text.lower())
    words = [word for word in text.split() if word not in stop_words]
    return ' '.join(words)


df['processed'] = df['message'].apply(preprocess)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['processed'])
y = df['label'].map({'ham': 0, 'spam': 1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

joblib.dump(model, 'spam_classifier_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("✅ Đã huấn luyện và lưu mô hình.")
