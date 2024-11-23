import torch
from transformers import BigBirdForSequenceClassification, BigBirdTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Load your dataset
df = pd.read_csv('reviews_dataset.csv')  # Replace with the path to your dataset

# Preprocessing: Tokenize the dataset
tokenizer = BigBirdTokenizer.from_pretrained('google/bigbird-roberta-base')

# Function to tokenize the input text
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

# Split dataset into train and test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Convert the dataframe into Hugging Face Dataset format
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Tokenize the datasets
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Map labels (0 for non-sarcastic, 1 for sarcastic)
train_dataset = train_dataset.map(lambda examples: {'label': examples['label']}, batched=True)
test_dataset = test_dataset.map(lambda examples: {'label': examples['label']}, batched=True)

# Load BigBirdForSequenceClassification model
model = BigBirdForSequenceClassification.from_pretrained('google/bigbird-roberta-base', num_labels=2)  # 2 labels: Sarcastic, Non-sarcastic

# Define TrainingArguments
training_args = TrainingArguments(
    output_dir='./sarcasm-results',  # Save results in a separate folder
    eval_strategy="epoch",  # Changed from evaluation_strategy to eval_strategy
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir='./sarcasm-logs',
    logging_steps=10,
    load_best_model_at_end=True,  # Load the best model at the end of training
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

try:
    # Train the model
    trainer.train()
except KeyboardInterrupt:
    print("Training interrupted. Saving model...")
finally:
    # Save the model
    model.save_pretrained('./bigbird-sarcasm-modelnew')  # Save the model with a different name
    tokenizer.save_pretrained('./bigbird-sarcasm-modelnew')
    print("Model saved successfully.")
