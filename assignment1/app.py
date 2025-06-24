import streamlit as st
import joblib

# Step 2: Set up the Basic App Layout
st.title('Streamlit Basics Tutorial')
st.markdown("This app will load a sentiment analysis classification model, allow a user to input a movie review, and use the classification model to output a predicted sentiment.")

# Step 3: Load the Saved Model
@st.cache_data
def load_model():
    """Loads the pre-trained model and target names."""
    model = joblib.load('sentiment_model.pkl')
    return model