import pandas as pd
import numpy as np
import torch
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# --- Configuration ---
MODEL_CHECKPOINT = "distilbert-base-uncased"
DATA_PATH_TRAIN = "./input/task-1-data/train.jsonl"
DATA_PATH_VAL = "./input/task-1-data/val.jsonl"
MODEL_OUTPUT_DIR = "./models/task1_distilbert"
os.environ["WANDB_DISABLED"] = "true"

# --- Dataset Loading and Preprocessing ---
class ClickbaitDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def load_data(path):
    df = pd.read_json(path, lines=True)
    df['text'] = df['postText'].apply(lambda x: x[0]) + " " + df['targetTitle']
    df['label_str'] = df['tags'].apply(lambda x: x[0])
    return df[['text', 'label_str']]

# --- Main Training Logic ---
def main():
    print("--- Starting Task 1: Model Training ---")

    # 1. Load and Prepare Data
    print("Loading and preparing data...")
    train_df = load_data(DATA_PATH_TRAIN)
    val_df = load_data(DATA_PATH_VAL)

    # Encode labels
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_df['label_str'])
    val_labels = label_encoder.transform(val_df['label_str'])
    num_labels = len(label_encoder.classes_)
    
    id2label = {i: label for i, label in enumerate(label_encoder.classes_)}
    label2id = {label: i for i, label in id2label.items()}

    # 2. Tokenize Text
    print("Tokenizing text...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    train_encodings = tokenizer(train_df['text'].tolist(), truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(val_df['text'].tolist(), truncation=True, padding=True, max_length=512)

    train_dataset = ClickbaitDataset(train_encodings, train_labels)
    val_dataset = ClickbaitDataset(val_encodings, val_labels)

    # 3. Define Model and Metrics
    print("Loading pre-trained model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        f1 = f1_score(p.label_ids, preds, average='weighted')
        acc = accuracy_score(p.label_ids, preds)
        return {'f1': f1, 'accuracy': acc}

    # 4. Set Training Arguments
    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to=[],
    )

    # 5. Initialize and Run Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    print("Starting fine-tuning...")
    trainer.train()

    # 6. Evaluate and Save
    print("Evaluating final model...")
    eval_results = trainer.evaluate()
    print(f"Validation Set Performance: {eval_results}")

    print("Saving the best model...")
    trainer.save_model(MODEL_OUTPUT_DIR)
    
    print("--- Training complete. Model saved to {} ---".format(MODEL_OUTPUT_DIR))

if __name__ == "__main__":
    main()