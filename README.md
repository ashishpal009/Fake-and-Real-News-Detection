# Fake and Real News Detection

## Abstract
With the increasing number of fraudulent websites on the internet, fake news has become a growing issue. Many websites profit through clickbait by publishing false information online. Users are often directed to pages containing fake news, which influences public perception and benefits content publishers financially. The rise of fake news has become a global challenge, even for major tech companies like Facebook and Google. Identifying whether a news article is real or fake without additional context is challenging and requires robust machine learning models.

---

## Problem Statement
The dataset consists of two folders containing **True** and **Fake** news articles. The objective is to develop a robust model to classify a given news article as **Real** or **Fake**. The project involves:

1. **Data Preparation**:
   - Cleaning data
   - Bag of Words (BoW)
   - Stemming
   - Lemmatization
   - Tokenization
   - MultinomialNB Algorithm
   - TF-IDF
   - Word vectorization (Word2Vec)
   - Splitting the dataset into training and testing sets

2. **Model Building**:
   - Developing a **Recurrent Neural Network (RNN)** model to classify news articles.

3. **Model Evaluation**:
   - Measuring accuracy, precision, recall, and F1-score.
   - Testing the model's generalization on unseen news articles.

---

## Scope of the Project

- Learn how to load and prepare text data for binary classification.
- Perform feature extraction from text data.
- Build a robust binary classifier for text classification problems.
- Classify new, unseen text messages as **Fake** or **Real News**.
- Gain hands-on experience with techniques like **Stemming**, **Lemmatization**, **MultinomialNB**, **Word2Vec**, and **TF-IDF**.


### Model Building
Build the RNN model using the `rnn_model.py` script or the notebook:

```bash
python src/rnn_model.py
```

### Model Evaluation
Evaluate the model's performance:

```bash
python src/evaluation.py
```

### 1. Data Cleaning
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Load data
data = pd.read_csv('news_dataset.csv')
data['label'] = data['label'].map({'Fake': 0, 'Real': 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['label'], test_size=0.2, random_state=42
)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
```

### 2. RNN Model
```python
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense

# Build the RNN model
model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=500),
    LSTM(units=128, return_sequences=False),
    Dense(units=1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
model.fit(X_train_tfidf, y_train, epochs=10, batch_size=32, validation_data=(X_test_tfidf, y_test))
```

### 3. Model Evaluation
```python
from sklearn.metrics import classification_report

# Predict on test data
y_pred = (model.predict(X_test_tfidf) > 0.5).astype(int)

# Evaluate the model
print(classification_report(y_test, y_pred))
```

---

## Key Features
- **Text Preprocessing**: TF-IDF, Word2Vec
- **Binary Classification**: Real vs Fake
- **Deep Learning**: RNN model for improved accuracy

---

## Performance Metrics
- Accuracy
- Precision
- Recall
- F1-score

---

## Contributors
- [Your Name](https://github.com/your-profile)

---

## License
This project is licensed under the MIT License.
