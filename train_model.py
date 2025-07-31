import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

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

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model pipeline: TF-IDF + Naive Bayes
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', MultinomialNB()),
])

# Train model
pipeline.fit(X_train, y_train)

# Evaluate on test data
y_pred = pipeline.predict(X_test)

# Print metrics
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Save model
joblib.dump(pipeline, 'model.pkl')
print("✅ Model trained and saved as 'model.pkl'")
