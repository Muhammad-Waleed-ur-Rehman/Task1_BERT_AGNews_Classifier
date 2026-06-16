# News Topic Classifier Using BERT

## Project Overview

This project demonstrates the application of Natural Language Processing (NLP) using Transformer-based models. A pre-trained BERT model (`bert-base-uncased`) was fine-tuned on the AG News dataset to classify news headlines into four categories:

- World
- Sports
- Business
- Sci/Tech

The project also includes a Streamlit web application that allows users to classify news headlines in real time.

---

# Problem Statement

News articles are generated in large volumes every day. Manually categorizing news content is time-consuming and inefficient. The objective of this project is to develop an automated news classification system using a transformer-based language model that can accurately classify news headlines into predefined categories.

---

# Objective

The primary objectives of this project are:

- Load and preprocess the AG News dataset.
- Fine-tune the BERT (`bert-base-uncased`) model for text classification.
- Evaluate model performance using Accuracy and F1-Score.
- Visualize classification results.
- Deploy the trained model using Streamlit for live predictions.

---

# Dataset

## AG News Dataset

**Source:** Hugging Face Datasets Library

The dataset contains news articles categorized into four classes:

| Label | Category |
|---------|------------|
| 0 | World |
| 1 | Sports |
| 2 | Business |
| 3 | Sci/Tech |

### Dataset Split

- Training Samples: 120,000
- Testing Samples: 7,600

For faster experimentation, a subset of the dataset was used during training.

---

# Technologies Used

## Programming Language

- Python

## Libraries and Frameworks

- Hugging Face Transformers
- Hugging Face Datasets
- PyTorch
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Streamlit

---

# Methodology

## Step 1: Dataset Loading

The AG News dataset was loaded using the Hugging Face Datasets library.

## Step 2: Data Preprocessing

The text data was tokenized using:

- BertTokenizerFast
- Maximum sequence length: 128
- Padding enabled
- Truncation enabled

## Step 3: Model Development

The pre-trained BERT model was loaded:

```python
bert-base-uncased
```

The classification head was modified to predict four output classes.

## Step 4: Fine-Tuning

The model was fine-tuned using the Hugging Face Trainer API with:

- Learning Rate: 2e-5
- Batch Size: 8
- Epochs: 2
- Weight Decay: 0.01

## Step 5: Model Evaluation

The model was evaluated using:

- Accuracy
- F1-Score


## Step 6: Deployment

The trained model was saved and deployed using Streamlit for real-time headline classification.

---

# Results

## Evaluation Metrics

The model was evaluated using:

- Accuracy Score
- F1 Score

### Example Prediction

**Input**

```text
Google unveils a new AI model for enterprise applications.
```

**Output**

```text
Predicted Category: Sci/Tech
Confidence Score: 98.91%
```

---

# Visualizations

The following visualizations were generated:

1. News Category Distribution

These visualizations helped analyze model.

---

# Streamlit Deployment

The project includes a Streamlit application for real-time interaction.

## Features

- User-friendly interface
- Live news headline classification
- Confidence score display
- Fast inference using the fine-tuned BERT model

### Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

# Project Structure

```text
TASK1_BERT_AGNEWS_CLASSIFIER/
│
├── saved_model/
├── .gitignore
├── app.py
├── README.md
├── requirements.txt
└── Task1_BERT_AGNews_Classifier.ipynb
```
---

# Key Results / Observations

The project successfully demonstrates how a pre-trained BERT model can be fine-tuned for news topic classification. The model effectively classifies news headlines into World, Sports, Business, and Sci/Tech categories while maintaining strong predictive performance. The Streamlit deployment further enables real-time interaction and practical use of the trained model.

---
