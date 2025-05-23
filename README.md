# Sentiment Analysis of Bangla Social Media Comments

## Project Overview

This project focuses on analyzing Bangla Facebook comments to detect sentiment, specifically classifying comments as **Bully** or **Not-Bully**. Using the publicly available **Bangla-Text-Dataset** of over 44,001 real-world social media comments, the goal is to build an accurate, lightweight model accessible via a FastAPI web service.

The workflow includes comprehensive text preprocessing, feature extraction using TF-IDF, training a Logistic Regression model, and deployment using ONNX for efficient inference. Experiments with advanced models like Bangla-BERT fine-tuning have also been conducted to explore accuracy improvements.

---

## Dataset Details

- **Source:** [Bangla-Text-Dataset on GitHub](https://github.com/cypher-07/Bangla-Text-Dataset/tree/main)
- **Content:** 44,001 Bangla Facebook comments from public pages (celebrities, politicians, athletes).
- **Labels:** Multiclass (`sexual`, `not bully`, `troll`, `religious`, `threat`)
- **Labeling:** Each comment is tagged with a single label, representing types of bullying or neutral sentiment.
- **Usage:** Build a model to classify bullying and general sentiment in Bangla social media comments.

---

### Dataset Gender Distribution

The dataset consists of the following gender distribution:

- **Female:** 68.07%
- **Male:** 31.93%


<div align="center">
  <img src="https://github.com/user-attachments/assets/859e2227-f7b1-41a6-a5f1-9226c02810be" alt="Image" />
</div>


---

### Dataset Multiclass Distribution 
**Multiclass** (`sexual`, `not bully`, `troll`, `religious`, `threat`)

<div align="center">
  <img src="https://github.com/user-attachments/assets/5f12f7f0-abad-4f66-b8c8-671003f801e8" alt="Image" />
</div>

---

## Technical Approach

### Required Libraries

- **pandas:** Manipulate and analyze data tables.
- **re (regular expressions):** Search and manipulate strings using patterns.
- **unicodedata:** Normalize and process Unicode text.
- **transformers (AutoTokenizer):** Tokenize text using pre-trained NLP models.
- **imblearn.over_sampling (SMOTE):** Balance dataset by creating synthetic samples of the minority class.
- **imblearn.under_sampling (RandomUnderSampler):** Reduce majority class to balance dataset.
- **sklearn.model_selection (train_test_split):** Split dataset into training and testing sets.
- **sklearn.feature_extraction.text (TfidfVectorizer):** Convert text to numerical features using TF-IDF.
- **sklearn.linear_model (LogisticRegression):** Train a classification model using logistic regression.
- **sklearn.metrics:** Evaluate model performance (accuracy, precision, recall, F1, confusion matrix, classification report).
- **random:** Perform random operations like shuffling and sampling.
- **matplotlib.pyplot:** Create visualizations and plots.
- **seaborn:** Enhanced statistical data visualization based on matplotlib.
- **joblib:** Save and load Python objects efficiently.

# Installation Instructions

To set up the required packages, run the following commands:

```bash
!pip install imbalanced-learn
!pip install skl2onnx onnxruntime
!pip install transformers

```
### 1. Data Preprocessing

- **Normalization:** The text is cleaned by removing digits, punctuations, web links, non-Bengali letters, and emojis, followed by Unicode normalization to maintain consistency across the dataset.
- **Stopword Removal:** Filter out common Bangla stopwords. [Bangla Stopwords Dataset](https://docs.google.com/spreadsheets/d/1bF6lGq1exiYNDOTSzsXd_TqxX3D5jKqg/edit?usp=sharing&ouid=114522712885813850468&rtpof=true&sd=true)
- **Tokenization:** I use the pretrained Bangla-BERT tokenizer (sagorsarker/bangla-bert-base) to break down the text into meaningful pieces called tokens. For instance, a sample normalized comment is split into smaller parts, and each part is then converted into a unique number the model can work with. This way, the raw text is transformed into a format that the model can easily understand and use.
- **Class Balancing:** Class imbalance is addressed by applying SMOTE to oversample minority classes and Random Under Sampling to reduce the majority class, creating a balanced dataset.

 <div align="center">
  <img src="https://github.com/user-attachments/assets/34a21046-dd37-4f72-96c7-8d1bcd2a1219" alt="Image" />
</div>


- **Label Encoding:** Labels are encoded in a binary format (Bully vs Not-Bully) for current model deployment, while multiclass encoding is available for future model extensions.

### 2. Feature Extraction

- Apply **TF-IDF Vectorization** to transform cleaned comments into numerical vectors representing term importance.


### 3. Model

#### 1. TF-IDF + Logistic Regression

This is a classic machine learning method where we first convert text into numbers using TF-IDF, which highlights how important each word is within a document compared to the whole dataset. These numbers are then passed to a Logistic Regression model that learns to separate different classes by drawing a decision boundary. It’s a simple and fast approach that works well for many text tasks but doesn’t fully capture the deeper meaning or context of the words.

#### 2. Bangla-BERT

Bangla-BERT is a modern deep learning model built specifically for the Bangla language. It understands language by looking at whole sentences at once and using attention mechanisms to grasp context and relationships between words. When fine-tuned for classification, Bangla-BERT usually performs much better because it can understand the meaning and nuances in the text. Although it needs more computing power, it’s great for handling complex language tasks where context really matters.


### 4. Model Training & Evaluation

- **Model:** Logistic Regression trained on TF-IDF features.
- **Performance Metrics:**

| Metric     | TF-IDF + Logistic Regression | Bangla-BERT Fine-tuning |
|------------|------------------------------|------------------------|
| Accuracy   | 0.6925                       | 0.8778                 |
| Precision  | 0.6835                       | 0.8737                 |
| Recall     | 0.5855                       | 0.8535                 |
| F1-Score   | 0.6307                       | 0.8635                 |

### Confusion Matrices

**TF-IDF + Logistic Regression**

|               | Predicted Not-Bully | Predicted Bully |
|---------------|---------------------|-----------------|
| Actual Not-Bully | 3784                | 1070            |
| Actual Bully     | 1636                | 2311            |


![Image](https://github.com/user-attachments/assets/41a6587d-b0e7-4586-b859-e18c6a1549fb)

---

**Bangla-BERT Fine-tuning**

|               | Predicted Not-Bully | Predicted Bully |
|---------------|---------------------|-----------------|
| Actual Not-Bully | 2162                | 246             |
| Actual Bully     | 292                 | 1701            |


![Image](https://github.com/user-attachments/assets/4b439389-e2f8-4471-9edb-7d4c9f527161)


---

These confusion matrices illustrate that Bangla-BERT greatly improves true positive and true negative classifications compared to the TF-IDF + Logistic Regression baseline.


# How to Run the API

## Code Organization
- I separated all necessary preprocessing functions such as text normalization and stopword removal into a dedicated Python file.  
- This file is imported into the main FastAPI app script.  
- The ONNX model file and stopwords dataset are placed in the same directory on the EC2 instance.  

## Virtual Environment Setup
- On the EC2 instance, I created a Python virtual environment (`venv`) to isolate dependencies.  
- This avoids version conflicts because different projects often require different package versions.  
- After creating the virtual environment, I activated it.  
- Inside the activated environment, I installed all required Python packages such as:  
  - `fastapi`  
  - `uvicorn`  
  - `onnxruntime`  
  - `pandas`  
  - `openpyxl`  
  - etc.  

## Running the FastAPI App
- With the virtual environment activated, I ran the FastAPI server using Uvicorn.  
- I specified the host as `0.0.0.0` to allow external access and the port as `8000`.  
- The server started successfully and was listening on port `8000`.  

## Testing the API with Postman
- I used Postman to test the API.  
- In Postman, I set the method to `POST` and the URL to `http://122.248.240.219:8000/predict`.  
- In the request body, I selected `raw` and set the format to `JSON`.  
- I wrote the JSON payload with a `"text"` field containing the Bangla comment to analyze.  
- After sending the request, I received the predicted label in the JSON response.  

## Example POST Request

```json
{
  "text": "তুমি খুব খারাপ একজন মানুষ"
}

```

## Example Predicted Label

```json
{
  "predicted_label": "Bully"
}

```

<div align="center">
  <img src="https://github.com/user-attachments/assets/d4d7da68-6914-41d3-8103-46defeb961e4" alt="Image" />
</div>

---

## Limitations and Future Improvements

- Cleaning the text even better, such as handling slang, emojis, and typos more accurately, can help the model understand the comments more clearly and improve results.  
- Advanced feature extraction techniques like GloVe, Word2Vec, or FastText can give the model a deeper understanding of the meaning behind each word, which can lead to better predictions.  
- Using fine-tuning on more powerful models like Bangla-BERT, Bangla DistilBERT, XLM-RoBERTa, or BanglaT5 could boost accuracy and help the system better understand complex language patterns.

---


## Note

Note: Although I encode the labels for both binary and multiclass classification, I focus only on the binary labels for both models (TF-IDF + Logistic Regression) and Bangla-BERT fine-tuning model. However, multiclass encoding has been done and can be applied to these models in future work. I have already applied encoding to multiclass in my code too, but I use binary.

Note: I have also worked on fine-tuning the Bangla-BERT model. However, both the Bangla-BERT model and its ONNX format are too large in size (exceeding 1GB) to upload to GitHub. Due to this file size limitation, I have only included the notebook file: `"Sentiment_Analysis_Bangla_BERT_Miftahul_Sheikh.ipynb"`.

Thank you for your understanding.


