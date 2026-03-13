import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import os

print("Starting AI Fake News Detector Training...")

# 1. Check if the files exist
true_path = 'True.csv'
fake_path = 'Fake.csv'

if not os.path.exists(true_path) or not os.path.exists(fake_path):
    print(f"ERROR: Cannot find {true_path} or {fake_path} in the current directory.")
    print("Please make sure you moved the downloaded files here!")
    exit()

print("Loading datasets... (this might take a minute)")
try:
    # 2. Load the data
    df_true = pd.read_csv(true_path)
    df_fake = pd.read_csv(fake_path)
except Exception as e:
    print(f"Error reading CSV files: {e}")
    print("If your files are .xlsx format, please export them as .csv first.")
    exit()

# 3. Add labels (1 for Fake, 0 for Real/True)
df_true['label'] = 0
df_fake['label'] = 1

# 4. Combine the datasets
df = pd.concat([df_true, df_fake], ignore_index=True)

# 5. Data preprocessing
print("Preprocessing text data...")
df.dropna(subset=['text'], inplace=True) # Remove empty rows

# A hidden issue in this dataset: True news often starts with "WASHINGTON (Reuters) - "
import re
def clean_text(text):
    # Remove the publisher prefix (e.g. "WASHINGTON (Reuters) - ")
    # The prefix usually ends with ' - '
    return re.sub(r'^.*? - ', '', text)

df['text'] = df['text'].apply(clean_text)

x = df['text']
y = df['label']

# 6. Split data into training and testing sets
print("Splitting dataset into Training (80%) and Testing (20%)...")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 7. Convert text to numbers (TF-IDF)
print("Vectorizing text (Converting words to numerical values with TF-IDF)...")
# Initialize the TfidfVectorizer with English stop words (removes 'the', 'is', 'in' etc.)
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform the training data, then transform the testing data
tfidf_train = tfidf_vectorizer.fit_transform(x_train) 
tfidf_test = tfidf_vectorizer.transform(x_test)

# 8. Define and Train the AI Model
print("Training the Passive-Aggressive Classifier Model...")
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

# 9. Evaluate the model
print("Evaluating the model on the test data...")
y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {round(score*100, 2)}%')
print(f'Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}')

# 10. Save the trained model and the vectorizer so we don't have to retrain it!
print("Saving the trained model to disk...")
with open('model.pkl', 'wb') as f:
    pickle.dump(pac, f)
    
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

print("Training Complete! You can now run 'python test.py' to test the AI.")
