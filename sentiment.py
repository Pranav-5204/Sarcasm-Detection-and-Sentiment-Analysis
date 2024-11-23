import torch
from transformers import BigBirdForSequenceClassification, BigBirdTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split

# Load your dataset
df = pd.read_csv('reviews.csv')  # Replace with the path to your dataset

# Preprocessing: Tokenize the dataset
tokenizer = BigBirdTokenizer.from_pretrained('google/bigbird-roberta-base')

# Function to tokenize the input text
def tokenize_function(examples):
    return tokenizer(examples['Review'], truncation=True, padding=True)

# Split dataset into train and test
train_df, test_df = train_test_split(df, test_size=0.2)

# Convert the dataframe into Hugging Face Dataset format
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Tokenize the datasets
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Specify the labels (convert sentiment to numerical)
label_dict = {'Positive': 0, 'Negative': 1, 'Neutral': 2}

train_dataset = train_dataset.map(lambda examples: {'label': [label_dict[sent] for sent in examples['Sentiment']]}, batched=True)
test_dataset = test_dataset.map(lambda examples: {'label': [label_dict[sent] for sent in examples['Sentiment']]}, batched=True)

# Load BigBirdForSequenceClassification model
model = BigBirdForSequenceClassification.from_pretrained('google/bigbird-roberta-base', num_labels=3)  # 3 labels: Positive, Negative, Neutral

# Define TrainingArguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir='./logs',
    logging_steps=10,
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained('./bigbird-sentiment-model')
tokenizer.save_pretrained('./bigbird-sentiment-model')
