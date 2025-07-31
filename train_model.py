# train_model.py
import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Dataset file
DATASET_FILE = 'UpdatedResumeDataSet.csv'
if not os.path.exists(DATASET_FILE):
    raise FileNotFoundError(f"Dataset file '{DATASET_FILE}' not found.")

# Load and clean dataset
df = pd.read_csv(DATASET_FILE)
df['Resume'] = df['Resume'].fillna('')
df['Category'] = df['Category'].fillna('Unknown')

# Features and labels
X = df['Resume']
y = df['Category']

# Model pipeline: TF-IDF + Naive Bayes
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', MultinomialNB()),
])

# Train and save
pipeline.fit(X, y)
joblib.dump(pipeline, 'model.pkl')
print("✅ Model trained and saved as 'model.pkl'")
