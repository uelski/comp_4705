import streamlit as st
import joblib

# Step 2: Set up the Basic App Layout
st.title('Streamlit Basics Tutorial')
st.markdown("This app will load a sentiment analysis classification model, allow a user to input a movie review, and use the classification model to output a predicted sentiment.")

# Step 3: Load the Saved Model
@st.cache_data
def load_model():
    """Loads the pre-trained model."""
    model = joblib.load('sentiment_model.pkl')
    return model

model = load_model()

# Step 4: Create the User Input Interface
user_text = st.text_area("Enter a movie review to analyze:")
analyze = st.button("Analyze")

# Step 5: Make Predictions and Display Results
if analyze:
    if len(user_text) > 0:
        output = model.predict([user_text])
        st.subheader(output)
        proba = model.predict_proba([user_text])
        st.subheader(proba)