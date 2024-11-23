'''import torch
from transformers import BigBirdForSequenceClassification, BigBirdTokenizer

# Load the trained model and tokenizer
model_path = 'bigbird-sentiment-model'  # Update this to your model's path
model = BigBirdForSequenceClassification.from_pretrained(model_path)
tokenizer = BigBirdTokenizer.from_pretrained(model_path)

# Set the model to evaluation mode
model.eval()

# Define a function to predict sentiment
def predict_sentiment(review):
    # Tokenize the input
    inputs = tokenizer(review, return_tensors="pt", padding=True, truncation=True)

    # Make the prediction
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    # Logits for analysis
    print(f"Logits: {logits}")

    # Mapping predicted class to sentiment label
    sentiment_labels = {0: 'Positive', 1: 'Negative', 2: 'Neutral'}
    return sentiment_labels[predicted_class]

# Example usage
if __name__ == "__main__":
    # Input your review here
    review = "Honestly, this self-help book is a complete waste of time! I mean, who wants to read about personal growth and improving their life? It’s not like I need to figure out how to be happier and more productive or anything. I definitely prefer to stay stuck in my old ways. Thanks for ruining my comfort zone!"
    
    # Make a prediction
    predicted_sentiment = predict_sentiment(review)
    print(f"Review: '{review}' | Predicted Sentiment: {predicted_sentiment}")
    '''
import torch
from transformers import BigBirdForSequenceClassification, BigBirdTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load the trained model and tokenizer
model_path = 'bigbird-sentiment-model'  # Update this to your model's path
model = BigBirdForSequenceClassification.from_pretrained(model_path)
tokenizer = BigBirdTokenizer.from_pretrained(model_path)

# Set the model to evaluation mode
model.eval()

# Define a function to predict sentiment
def predict_sentiment(review):
    # Tokenize the input
    inputs = tokenizer(review, return_tensors="pt", padding=True, truncation=True)

    # Make the prediction
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    # Logits for analysis
    print(f"Logits: {logits}")

    # Mapping predicted class to sentiment label
    sentiment_labels = {0: 'Positive', 1: 'Negative', 2: 'Neutral'}
    return predicted_class, sentiment_labels[predicted_class]  # Return both the predicted class and the sentiment label

# Example usage
if __name__ == "__main__":
    # List of example reviews and their true labels (for evaluation)
    reviews = [
        "Honestly, this self-help book is a complete waste of time! I mean, who wants to read about personal growth and improving their life? It’s not like I need to figure out how to be happier and more productive or anything. I definitely prefer to stay stuck in my old ways. Thanks for ruining my comfort zone!",
        "This was the best self-help book I've ever read! I feel so inspired and motivated to improve myself.",
        "Oh, it’s just perfect—if you enjoy waiting for something to break after a single use!"
    ]
    true_labels = [1, 0, 2]  # The true sentiment labels corresponding to the reviews (0=Positive, 1=Negative, 2=Neutral)

    # Store predictions and true labels for evaluation
    predictions = []
    pred_labels = []

    for review in reviews:
        predicted_class, sentiment = predict_sentiment(review)
        predictions.append(predicted_class)  # Store the predicted class
        pred_labels.append(sentiment)  # Store the predicted label
        print(f"Review: '{review}' | Predicted Sentiment: {sentiment}")

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='macro')

    # Print metrics
    print("\nMetrics:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")
