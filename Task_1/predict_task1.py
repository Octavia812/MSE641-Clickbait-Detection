import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# --- Configuration ---
DATA_PATH_TEST = "./input/task-1-data/test.jsonl"
MODEL_PATH = "./models/task1_distilbert/"
SUBMISSION_PATH = "./submission_task1.csv"

# --- Main Prediction Logic ---
def main():
    print("--- Starting Task 1: Prediction ---")

    # 1. Load Test Data
    print("Loading test data...")
    test_df = pd.read_json(DATA_PATH_TEST, lines=True)
    test_df['text'] = test_df['postText'].apply(lambda x: x[0]) + " " + test_df['targetTitle']
    
    # Convert to Hugging Face Dataset
    test_dataset = Dataset.from_pandas(test_df)

    # 2. Load Model and Tokenizer
    print(f"Loading fine-tuned model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    # 3. Tokenize Data
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    print("Tokenizing test data...")
    tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

    # 4. Generate Predictions
    print("Generating predictions...")
    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir="./temp_predictions", per_device_eval_batch_size=16)
    )
    
    predictions = trainer.predict(tokenized_test_dataset)
    predicted_indices = np.argmax(predictions.predictions, axis=1)
    
    # Map indices back to labels
    id2label = model.config.id2label
    predicted_labels = [id2label[i] for i in predicted_indices]

    # 5. Create Submission File
    print(f"Creating submission file at {SUBMISSION_PATH}...")
    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'spoilerType': predicted_labels
    })

    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print("--- Prediction complete. ---")

if __name__ == "__main__":
    main()
