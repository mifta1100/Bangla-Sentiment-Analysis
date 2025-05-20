Project overview: This project focuses on analyzing Bangla Facebook comments to determine the sentiment behind them, whether it is Bully or Not-Bully. I used a publicly available dataset called the Bangla-Text-Dataset collected from the GitHub, which contains over 44,001 real comments collected from social media. The goal is to build a system that can accurately classify each comment's sentiment and make it accessible through a simple web API built with FastAPI. Along the way, I handled everything from cleaning the data and training the model to converting it into a ONNX format suitable for deployment. I preprocessed the data, Remove Stopwords, Use Pre-trained models like Bangla-BERT for Tokenize, and then Encode to Binary. I Apply Logistic Regression Model with TF-IDF feature extraction technique. During the project, I experimented with multiple modeling approaches, including BiLSTM and Bangla-BERT, to explore how well different architectures perform on Bangla sentiment classification. However, for deployment purposes, I use the Logistic Regression model with TF-IDF features in the FastAPI endpoint.


|-----------------------------------------------------------------------------------------------------------------------|


Dataset Resource and Purpose:

The dataset is sourced from the Bangla-Text-Dataset on GitHub. It contains 44,001 social media comments scraped from public Facebook pages, including those of celebrities, politicians, and athletes. Each comment is labeled either with bullying categories or sentiment labels.

This dataset is used to build a Bangla sentiment analysis model that can understand and sort social media comments into different feelings or categories. It helps to see what people think and feel about different topics on Bangla social media. The model can also spot bullying comments and keep track of conversations online. In the end, the model will be available as an easy-to-use FastAPI service that can classify comments quickly in real-time.

Dataset Review: The dataset consists of social media comments written in Bangla, which serve as the input texts. Each comment represents an individual piece of text expressing opinions, reactions, or statements related to celebrities, politicians, athletes, or social issues.

Comment: The comment column holds the actual text from social media that will be analyzed or classified.

Label: The label column shows the category of each comment. such as:

sexual
not bully
troll
religious
threat

label structure: The dataset uses a multiclass label structure because the comments are categorized into different types of bullying or sentiments. This means that each comment is labeled with one specific category, such as offensive language, hate speech, neutral, positive, or negative.


|-----------------------------------------------------------------------------------------------------------------------|


Result Analysis: 

TF-IDF + Logistic Regression: 
Accuracy:  0.6925
Precision: 0.6835
Recall:    0.5855
F1-score:  0.6307

Confusion Matrix:
[[3784 1070]
 [1636 2311]]

Classification Report:
              precision    recall  f1-score   support

           0       0.70      0.78      0.74      4854
           1       0.68      0.59      0.63      3947

    accuracy                           0.69      8801
   macro avg       0.69      0.68      0.68      8801
weighted avg       0.69      0.69      0.69      8801


Bangla-BERT:

Accuracy:  0.8778
Precision: 0.8737
Recall:    0.8535
F1-score:  0.8635
Confusion Matrix:
[[2162  246]
 [ 292 1701]]
(0.8777550556691661,
 0.8736517719568567,
 0.8534872052182639,
 0.8634517766497461,
 array([[2162,  246],
        [ 292, 1701]]))


|-----------------------------------------------------------------------------------------------------------------------|


How to Run the API:

Code organization:
1. I separated all necessary preprocessing functions such as: text normalization, stopword removal into a dedicated Python file.
2. This file is imported into the main FastAPI app script.
3. The ONNX model file and stopwords dataset are placed in the same directory on the EC2 instance.

Virtual Environment Setup:
1. On the EC2 instance, I created a Python virtual environment (venv) to isolate dependencies.
2. This avoids version conflicts because different projects often require different package versions.
3. After creating the virtual environment, I activated it.
4. Inside the activated environment, I installed all required Python packages such as: fastapi, uvicorn, onnxruntime, pandas, openpyxl, etc.

Running the FastAPI app:
1. With the virtual environment activated, I ran the FastAPI server using Uvicorn.
2. I specified the host as 0.0.0.0 to allow external access and the port as 8000.
3. The server started successfully and was listening on port 8000.

Testing the API with Postman:
1. I used Postman to test the API.
2. In Postman, I set the method to POST and the URL to http://122.248.240.219:8000/predict.
3. In the request body, I selected raw and set the format to JSON.
4. I wrote the JSON payload with a "text" field containing the Bangla comment to analyze.
5. After sending the request, I received the predicted label in the JSON response.


Example POST Request:

1. In POSTMAN, use the http://122.248.240.219:8000/predict
2. Content-Type: application/json

3. {
  "text": "তুমি খুব খারাপ একজন মানুষ"
}

4. {
  "predicted_label": "Bully"
}


|-----------------------------------------------------------------------------------------------------------------------|


Limitations and future improvements: 
1. Cleaning the text even better like handling slang, emojis, and typos more accurately, can help the model understand the comments more clearly and improve results.
2. Advanced Feature extraction technique like GloVe, Word2Vec, or FastText can give the model a deeper understanding of the meaning behind each word, which can lead to better predictions.
3. Using fine-tuning more powerful models like Bangla-BERT, Bangla DistilBERT, XLM-RoBERTa, or BanglaT5 could boost accuracy and help the system better understand complex language patterns.


|-----------------------------------------------------------------------------------------------------------------------|


Note: Although I encode the labels for both binary and multiclass classification, I focus only on the binary labels for both model (TF-IDF + Logistic Regression) and Bangla-BERT fine-tuning model. However, multiclass encoding has been done and can be applied to these models in future work. I have already apply Encoding to multiclass in my code too but I use Binary.


Note: I have also worked on fine-tuning the Bangla-BERT model. However, both the Bangla-BERT model and its ONNX format are too large in size (exceeding 1GB) to upload to GitHub. Due to this file size limitation, I have only included the notebook file: "Sentiment_Analysis_Bangla_BERT_Miftahul_Sheikh.ipynb". Thank you for your understanding.


|-----------------------------------------------------------------------------------------------------------------------|
