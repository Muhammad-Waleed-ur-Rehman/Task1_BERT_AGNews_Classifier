import streamlit as st
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification

# Page configuration
st.set_page_config(
    page_title="News Topic Classifier",
    page_icon="📰",
    layout="centered"
)

# Label mapping
label_names = ["World", "Sports", "Business", "Sci/Tech"]

# Load model and tokenizer
@st.cache_resource
def load_model():
    model_path = "saved_model"
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# Prediction function
def predict_news_topic(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

    return label_names[predicted_class], confidence, probabilities[0].tolist()

# App UI
st.title("📰 News Topic Classifier Using BERT")

st.write(
    "This app classifies news headlines into one of four categories: "
    "**World, Sports, Business, or Sci/Tech**."
)

headline = st.text_area(
    "Enter a news headline:",
    placeholder="Example: Microsoft announces new AI tools for business users"
)

if st.button("Classify"):
    if headline.strip() == "":
        st.warning("Please enter a news headline.")
    else:
        category, confidence, probabilities = predict_news_topic(headline)

        st.success(f"Predicted Category: **{category}**")
        st.info(f"Confidence Score: **{confidence:.2%}**")

        st.subheader("Category Probabilities")

        prob_data = {
            "Category": label_names,
            "Probability": probabilities
        }

        st.bar_chart(
            data={
                label_names[i]: probabilities[i]
                for i in range(len(label_names))
            }
        )

st.markdown("---")
st.caption("Model: Fine-tuned bert-base-uncased on AG News Dataset")