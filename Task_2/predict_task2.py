import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm

# --- Configuration ---
DATA_PATH_TEST = "./input/task-2-data/test.jsonl"
MODEL_PATH = "./models/task2_t5_base/"
SUBMISSION_PATH = "./submission_task2.csv"
PREFIX = "spoil this headline: "
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Main Prediction Logic ---
def main():
    print("--- Starting Task 2: Prediction ---")
    print(f"Using device: {DEVICE}")

    # 1. Load Test Data
    print("Loading test data...")
    test_df = pd.read_json(DATA_PATH_TEST, lines=True)
    test_df['input_text'] = test_df.apply(
        lambda row: PREFIX + row['postText'][0] + " context: " + " ".join(row['targetParagraphs']),
        axis=1
    )

    # 2. Load Fine-tuned Model and Tokenizer
    print(f"Loading fine-tuned model from {MODEL_PATH}...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH).to(DEVICE)
    model.eval()

    # 3. Generate Predictions
    print("Generating spoilers for the test set...")
    predictions = []
    with torch.no_grad():
        for text in tqdm(test_df['input_text'], desc="Generating"):
            inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True).to(DEVICE)
            output_sequences = model.generate(
                **inputs,
                max_new_tokens=128,
                num_beams=5,
                early_stopping=True
            )
            prediction = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
            predictions.append(prediction)

    # 4. Create Submission File
    print(f"Creating submission file at {SUBMISSION_PATH}...")
    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'spoiler': predictions
    })

    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print("--- Prediction complete. ---")

if __name__ == "__main__":
    main()
