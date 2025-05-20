from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import onnxruntime as ort
import unicodedata


import pandas as pd

# Load Bangla stopwords
stopwords_df = pd.read_excel('Stopwords.xlsx')
stopwords_list = stopwords_df.iloc[:, 0].astype(str).tolist()


def preprocess_input(text):
    text = normalize_bangla_text(text)
    text = remove_stopwords(text)
    text = unicode_normalize(text)
    return text


# Define input data model for POST request
class Comment(BaseModel):
    text: str

app = FastAPI(title="Bangla Comment Sentiment Prediction")

# Load ONNX model session globally
onnx_session = ort.InferenceSession("bangla_logistic_regression.onnx")

def unicode_normalize(text):
    return unicodedata.normalize('NFC', text)

def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word not in stopwords_list]
    return ' '.join(filtered_words)

def normalize_bangla_text(text):
    # Convert to string in case of NaN
    text = str(text)

    # Remove web links
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove digits
    text = re.sub(r'\d+', '', text)

    # Remove punctuation and symbols
    text = re.sub(r'[^\u0980-\u09FF\s]', '', text)

    # Remove English letters
    text = re.sub(r'[a-zA-Z]', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# Helper function: preprocess text
def preprocess_text(text):
    processed_text = preprocess_input(text) 
    return processed_text

# Endpoint to accept comment and return predicted label
@app.post("/predict")
async def predict_sentiment(comment: Comment):
    try:
        # Preprocess input
        processed_text = preprocess_text(comment.text)

        # Prepare input for ONNX model
        input_name = onnx_session.get_inputs()[0].name
        inputs = {input_name: [[processed_text]]}  # shape (1,1)

        # Run inference
        pred_onx = onnx_session.run(None, inputs)
        pred_label = int(pred_onx[0][0])  

        # Map label id to text 
        label_map = {0: "Not Bully", 1: "Bully"}

        return {"predicted_label": label_map.get(pred_label, "Unknown")}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))