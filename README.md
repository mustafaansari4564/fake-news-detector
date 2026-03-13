# AI Fake News Detector 🤖📰

An Artificial Intelligence model built with Python and `scikit-learn` that uses Natural Language Processing (NLP) to detect whether a news article is real or fake.

## Features
- Achieved **98.65% Accuracy** using a Passive-Aggressive Classifier.
- Uses **TF-IDF Vectorization** to process text streams.
- Trained on a dataset of over 40,000 news articles.
- Eliminates "publisher bias" from the dataset for genuine pattern learning.
- Includes a fast, interactive command-line interface for testing new articles.

## How to Run Locally

### Prerequisites
You need Python installed. First, clone this repository and navigate into the folder:
```bash
git clone https://github.com/mustafaansari4564/fake-news-detector.git
cd fake-news-detector
```

### Installation
Install the required machine learning dependencies:
```bash
pip install -r requirements.txt
```

### Testing the AI
Since the pre-trained `model.pkl` and `vectorizer.pkl` are ignored by git (due to large file sizes), you must run the training script once to generate the AI brain:

1. Download the `True.csv` and `Fake.csv` dataset files (e.g., from Kaggle).
2. Place both CSV files in the root folder of this project.
3. Train the model:
   ```bash
   python train.py
   ```
4. Once training completes, run the detector script to verify any news article:
   ```bash
   python test.py
   # Or double-click run_detector.bat on Windows
   ```

## Tech Stack
- Python
- pandas (Data Manipulation)
- scikit-learn (Machine Learning & NLP)
- numpy
