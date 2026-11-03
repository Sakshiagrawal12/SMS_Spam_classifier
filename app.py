
import streamlit as st
import joblib
import os

from text_preprocessing import transform_text

MODEL_PATH = "artifacts/spam_classifier.pkl"

st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📩",
    layout="centered"
)


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None

    return joblib.load(MODEL_PATH)


artifact = load_model()

st.title("📩 SMS Spam Classifier")
st.write("Enter an SMS message to check whether it is Spam or Ham.")

if artifact is None:
    st.error("Model not found. Please run train_model.py first.")
    st.stop()

model = artifact["model"]
vectorizer = artifact["vectorizer"]
metrics = artifact["metrics"]

st.subheader("Model Performance")

col1, col2 = st.columns(2)

col1.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
col2.metric("F1 Score", f"{metrics['f1_score'] * 100:.2f}%")

message = st.text_area(
    "Enter your SMS message",
    placeholder="Example: Congratulations! You have won a free prize..."
)

if st.button("Check Message", type="primary"):

    if not message.strip():
        st.warning("Please enter a message first.")

    else:
        processed_message = transform_text(message)

        if not processed_message.strip():
            st.warning("The message does not contain usable text.")
        else:
            message_vector = vectorizer.transform([processed_message])

            prediction = model.predict(message_vector)[0]
            probabilities = model.predict_proba(message_vector)[0]

            spam_probability = probabilities[1] * 100

            if prediction == 1:
                st.error("🚨 This message is SPAM")
            else:
                st.success("✅ This message is HAM (Not Spam)")

            st.write(
                f"Spam probability: **{spam_probability:.2f}%**"
            )
