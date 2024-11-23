import pandas as pd
import torch
from transformers import pipeline

# Load your models
sentiment_model = pipeline("text-classification", model="bigbird-sentiment-model")  # Replace with your sentiment model
sarcasm_model = pipeline("text-classification", model="bigbird-sarcasm-modelnew")  # Replace with your sarcasm model

# Load the Excel file
df = pd.read_excel('SamsungFridgePrdtReviews.xlsx')  # Replace with your file name
df['Date'] = pd.to_datetime(df['Date'])  # Convert the Date column to datetime
df.sort_values('Date', inplace=True)  # Sort by Date

# Initialize lists for new columns
sentiment = []
sarcasm = []
sentiment_score = []
sarcasm_score = []
result = []

# Analyze each review
for review in df['Review']:
    # Sentiment analysis
    sentiment_output = sentiment_model(review)[0]
    sentiment_label = sentiment_output['label']
    sentiment_score_value = sentiment_output['score']

    # Sarcasm detection
    sarcasm_output = sarcasm_model(review)[0]
    sarcasm_label = sarcasm_output['label']
    sarcasm_score_value = sarcasm_output['score']

    # Map sentiment labels to descriptive strings
    if sentiment_label == 'LABEL_0':  # Positive
        sentiment_label = "Positive"
    elif sentiment_label == 'LABEL_1':  # Negative
        sentiment_label = "Negative"
    elif sentiment_label == 'LABEL_2':  # Neutral
        sentiment_label = "Neutral"

    sarcasm_label = "Sarcastic" if sarcasm_label == 'LABEL_1' else "Non-sarcastic"  # Descriptive labels

    # Determine the result based on sarcasm
  #  if sarcasm_label == "Sarcastic":  # If sarcastic
   #     result_label = "Negative" if sentiment_label == "Positive" else "Positive" if sentiment_label == "Negative" else "Neutral"
    #else:  # If non-sarcastic
    result_label = sentiment_label

    # Append results
    sentiment.append(sentiment_label)
    sarcasm.append(sarcasm_label)
    sentiment_score.append(sentiment_score_value)
    sarcasm_score.append(sarcasm_score_value)
    result.append(result_label)

# Create a new DataFrame with the results
results_df = pd.DataFrame({
    'Date': df['Date'],
    'Review': df['Review'],
    'Sentiment': sentiment,
    'Sarcasm': sarcasm,
    'Sentiment Score': sentiment_score,
    'Sarcasm Score': sarcasm_score,
    'Result': result
})

# Save the results to a new Excel file
results_df.to_excel('fridgereviewssorted.xlsx', index=False)

print("Processing complete. Results saved '.")
