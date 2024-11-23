'''import torch
from transformers import BigBirdForSequenceClassification, BigBirdTokenizer

# Load the trained model and tokenizer
model_path = './bigbird-sarcasm-modelnew'  # Path to your saved model
model = BigBirdForSequenceClassification.from_pretrained(model_path)
tokenizer = BigBirdTokenizer.from_pretrained(model_path)

# Function to check if a sentence is sarcastic
def is_sarcastic(sentence):
    # Tokenize the input sentence
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding='max_length', max_length=512)

    # Make sure to use the appropriate device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    # Move input tensors to the same device as the model
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Get the model's prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get predicted label
    predicted_label = torch.argmax(logits, dim=-1).item()

    # Map label to meaning
    return "Sarcastic" if predicted_label == 1 else "Not Sarcastic"

# Example usage
sentence = "Wow, I just love how this phone's battery dies after only two hours—exactly what I needed in my life!"  # Replace with your own sentence
result = is_sarcastic(sentence)
print(f'The sentence: "{sentence}" is {result}.')
'''

import torch
from transformers import BigBirdForSequenceClassification, BigBirdTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the trained model and tokenizer
model_path = './bigbird-sarcasm-modelnew'  # Path to your saved model
model = BigBirdForSequenceClassification.from_pretrained(model_path)
tokenizer = BigBirdTokenizer.from_pretrained(model_path)

# Function to check if a sentence is sarcastic
def is_sarcastic(sentence):
    # Tokenize the input sentence
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding='max_length', max_length=512)

    # Make sure to use the appropriate device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    # Move input tensors to the same device as the model
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Get the model's prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get predicted label
    predicted_label = torch.argmax(logits, dim=-1).item()

    # Map label to meaning
    return "Sarcastic" if predicted_label == 1 else "Not Sarcastic"

# Example product reviews
test_reviews = [
    "I absolutely love how this phone's battery dies so quickly—only two hours of use! Just what I needed from a phone—such a disappointment.",
    "Sleek and comfortable to use, with soft bristles that feel great on gums. However, the battery life barely lasts a few uses, and the repetitive music feature is more annoying than helpful. Not terrible, but definitely not groundbreaking.",
    "Wow, where do I even start? This product is revolutionary. Who knew I needed a toothbrush that plays elevator music while brushing? Just what I wanted – every morning feels like I’m brushing my teeth in the waiting room of a doctor’s office.",
    "This pair of running shoes made my walk in the park so much more enjoyable! They were super comfortable, and I felt so relaxed during the entire walk."
]

# True labels for sarcasm detection (0 = Not Sarcastic, 1 = Sarcastic)
true_labels = [1, 0, 1, 0]  # True labels for the reviews

# Predict sarcasm for each review
predicted_labels = [is_sarcastic(review) for review in test_reviews]

# Map "Sarcastic" to 1 and "Not Sarcastic" to 0 for metric calculation
predicted_labels_numeric = [1 if label == "Sarcastic" else 0 for label in predicted_labels]

# Calculate the evaluation metrics
accuracy = accuracy_score(true_labels, predicted_labels_numeric)
precision = precision_score(true_labels, predicted_labels_numeric)
recall = recall_score(true_labels, predicted_labels_numeric)
f1 = f1_score(true_labels, predicted_labels_numeric)

# Print results
for i, review in enumerate(test_reviews):
    print(f"Review: '{review}' | Predicted Sentiment: {predicted_labels[i]}")

print("\nEvaluation Metrics:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")
