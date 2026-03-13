import pickle
import sys

print("Loading the trained Fake News AI Model...")

try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
except FileNotFoundError:
    print("ERROR: Could not find the trained model files (model.pkl and vectorizer.pkl).")
    print("Please run 'python train.py' first to train the AI.")
    sys.exit()

print("\nModels loaded successfully! You can type 'quit' to exit.")

while True:
    print("\n---------------------------------------------------------")
    news_text = input("Paste a paragraph of news text to verify: \n")
    
    if news_text.lower() == 'quit':
        print("Goodbye!")
        break
    
    if len(news_text.strip()) < 10:
        print("Please enter a longer piece of text for accurate detection.")
        continue

    # Vectorize the input text using the PRE-TRAINED vectorizer
    vectorized_text = vectorizer.transform([news_text])
    
    # Predict using the loaded model
    prediction = model.predict(vectorized_text)
    
    if prediction[0] == 1:
        print("\n🚨🚨 RESULT: The AI predicts this news is FAKE. 🚨🚨")
    else:
        print("\n✅✅ RESULT: The AI predicts this news is REAL/TRUE. ✅✅")
