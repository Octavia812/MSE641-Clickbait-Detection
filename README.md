# MSE 641 – Clickbait Detection Challenge

This repository hosts all code and documentation for the **MSE 641 course project**, focused on the SemEval-2023 **Clickbait Spoiling** challenge. The project is split into two main tasks:

| Task | Objective | Evaluation Metric | Base Model |
|------|-----------|------------------|------------|
| **Task 1** | Classify clickbait posts into three spoiler types: `phrase`, `passage`, or `multi` | Weighted F1 | `distilbert-base-uncased` |
| **Task 2** | Generate a short text that "spoils" the clickbait and satisfies readers' curiosity | METEOR | `t5-base` |

---

## Directory Structure

```
/MSE641-Clickbait-Detection
├── README.md
├── requirements.txt
├── input/              
│   ├── task-1-data/
│   └── task-2-data/
├── Task_1/
│   ├── train_task1.py
│   └── predict_task1.py
└── Task_2/
    ├── train_task2.py
    └── predict_task2.py
```

---

## Environment Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# 2. (Recommended) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows, use venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Task 2 only) Download NLTK resources
python -c "
import nltk
for pkg in ['wordnet', 'punkt', 'omw-1.4']:
    nltk.download(pkg)
"
```

---

## Downloading and Placing the Datasets

> **Note:** The Kaggle datasets are large and therefore not included in this repository. Please download them manually.

| Task   | Kaggle Link                                                                                                                                                            |
| ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Task 1 | [https://www.kaggle.com/competitions/task-1-clickbait-detection-msci-641-s-25/data](https://www.kaggle.com/competitions/task-1-clickbait-detection-msci-641-s-25/data) |
| Task 2 | [https://www.kaggle.com/competitions/task-2-clickbait-detection-msci-641-s-25/data](https://www.kaggle.com/competitions/task-2-clickbait-detection-msci-641-s-25/data) |

After downloading, create the following folder structure and place the `train` / `val` / `test` `.jsonl` files accordingly:

```
/MSE641-Clickbait-Detection/input/
├── task-1-data/
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
└── task-2-data/
    ├── train.jsonl
    ├── val.jsonl
    └── test.jsonl
```

---

## How to Run

### Task 1: Spoiler Type Classification

```bash
# 1. Train
python Task_1/train_task1.py

# 2. Predict and create the submission file
python Task_1/predict_task1.py

# Output: ./submission_task1.csv
```

### Task 2: Spoiler Generation

```bash
# 1. Train
python Task_2/train_task2.py

# 2. Predict and create the submission file
python Task_2/predict_task2.py

# Output: ./submission_task2.csv
```

---

## Project Details

### Task 1: Classification Model
- **Base Model**: DistilBERT (distilbert-base-uncased)
- **Objective**: Multi-class classification of clickbait spoiler types
- **Classes**: 
  - `phrase`: Short answer or key information
  - `passage`: Longer explanatory text
  - `multi`: Multiple pieces of information needed

### Task 2: Text Generation Model
- **Base Model**: T5 (t5-base)
- **Objective**: Generate spoiler text that satisfies curiosity about clickbait
- **Evaluation**: METEOR score for text quality and relevance

---

## Requirements

Key dependencies include:
- `transformers`
- `torch`
- `datasets`
- `nltk`
- `pandas`
- `numpy`
- `scikit-learn`

See `requirements.txt` for complete list and versions.

---

## Results and Evaluation

Models are evaluated using:
- **Task 1**: Weighted F1-score across all classes
- **Task 2**: METEOR score for generated text quality

Final submission files (`submission_task1.csv` and `submission_task2.csv`) are generated for Kaggle competition submission.
