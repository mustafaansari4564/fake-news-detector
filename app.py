import streamlit as st
import pickle

st.set_page_config(page_title="AI Fake News Detector", page_icon="🤖", layout="centered")

# Custom CSS for a slight neon/hacker vibe
st.markdown("""
    <style>
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    .title {
        text-align: center;
        color: #00f3ff;
        font-family: 'Courier New', Courier, monospace;
    }
    .subtitle {
        text-align: center;
        color: #ff00ea;
        margin-bottom: 30px;
    }
    .result-fake {
        color: #ff4b4b;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        border: 2px solid #ff4b4b;
        border-radius: 10px;
        background-color: rgba(255, 75, 75, 0.1);
    }
    .result-real {
        color: #00fa9a;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        border: 2px solid #00fa9a;
        border-radius: 10px;
        background-color: rgba(0, 250, 154, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title'>🤖 AI Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("<h3 class='subtitle'>Analyze news text using Machine Learning</h3>", unsafe_allow_html=True)

# Function to load models with caching so it doesn't reload on every button click
@st.cache_resource
def load_models():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None

model, vectorizer = load_models()

st.write("Paste the text of a news article below to see if the AI thinks it's real or fake.")

user_input = st.text_area("News Article Text:", height=250, placeholder="Paste article here...")

if st.button("Detect with AI 🔍"):
    if not user_input or len(user_input.strip()) < 20:
        st.warning("Please enter a longer piece of text for accurate detection.")
    elif model and vectorizer:
        with st.spinner("AI is analyzing text..."):
            # Vectorize the input
            vectorized_text = vectorizer.transform([user_input])
            
            # Predict
            prediction = model.predict(vectorized_text)
            
            st.markdown("---")
            if prediction[0] == 1:
                st.markdown("<div class='result-fake'>🚨 AI Prediction: FAKE NEWS 🚨</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='result-real'>✅ AI Prediction: REAL NEWS ✅</div>", unsafe_allow_html=True)
    else:
        st.error("Model not loaded properly.")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #8b9bb4;'>Powered by Natural Language Processing and Scikit-Learn</p>", unsafe_allow_html=True)
