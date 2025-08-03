import pandas as pd
import numpy as np
import os
import evaluate
from datasets import Dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)

# --- Configuration ---
MODEL_CHECKPOINT = "t5-base"
DATA_PATH_TRAIN = "./input/task-2-data/train.jsonl"
DATA_PATH_VAL = "./input/task-2-data/val.jsonl"
MODEL_OUTPUT_DIR = "./models/task2_t5_base"
PREFIX = "spoil this headline: "

# --- Helper Functions ---
def preprocess_function(df):
    inputs = [PREFIX + post[0] + " context: " + " ".join(paragraphs) 
              for post, paragraphs in zip(df['postText'], df['targetParagraphs'])]
    targets = [spoiler[0] for spoiler in df['spoiler']]
    
    processed_df = pd.DataFrame({
        'input_text': inputs,
        'target_text': targets
    })
    return Dataset.from_pandas(processed_df)

# --- Main Training Logic ---
def main():
    print("--- Starting Task 2: Model Training ---")

    # 1. Load and Preprocess Data
    print("Loading and preprocessing data...")
    train_df = pd.read_json(DATA_PATH_TRAIN, lines=True)
    val_df = pd.read_json(DATA_PATH_VAL, lines=True)

    train_dataset = preprocess_function(train_df)
    val_dataset = preprocess_function(val_df)

    # 2. Load Model and Tokenizer
    print(f"Loading pre-trained model: {MODEL_CHECKPOINT}...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_CHECKPOINT)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_CHECKPOINT)

    # 3. Tokenize Datasets
    print("Tokenizing datasets...")
    def tokenize_map_function(examples):
        model_inputs = tokenizer(examples['input_text'], max_length=1024, truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples['target_text'], max_length=128, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_train_ds = train_dataset.map(tokenize_map_function, batched=True)
    tokenized_val_ds = val_dataset.map(tokenize_map_function, batched=True)

    # 4. Set up Trainer
    print("Setting up trainer...")
    meteor = evaluate.load("meteor")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = meteor.compute(predictions=decoded_preds, references=decoded_labels)
        return {"meteor": result["meteor"]}

    training_args = Seq2SeqTrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=2,
        weight_decay=0.01,
        save_total_limit=2,
        predict_with_generate=True,
        fp16=True,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="meteor",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_ds,
        eval_dataset=tokenized_val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 5. Start Training
    print("Starting T5 fine-tuning...")
    trainer.train()

    # 6. Save Model
    print("Saving the best model...")
    trainer.save_model(MODEL_OUTPUT_DIR)

    print("--- Training complete. Model saved to {} ---".format(MODEL_OUTPUT_DIR))

if __name__ == "__main__":
    main()
