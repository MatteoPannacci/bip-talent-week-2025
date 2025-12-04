import os
import re
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset

from utilities import *

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed
)



os.environ["TOKENIZERS_PARALLELISM"] = "false"



# Constants

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(MAIN_DIR, "data/data_evaluation.xlsx")
OUTPUT_PATH = os.path.join(MAIN_DIR, "output_pred_matteo_pannacci.xlsx")
MODEL_DIR = os.path.join(MAIN_DIR, "model")



# Find last checkpoint

all_checkpoints = [d for d in os.listdir(MODEL_DIR) if os.path.isdir(os.path.join(MODEL_DIR, d))]

checkpoint_pattern = re.compile(r"checkpoint-(\d+)")
checkpoints = [(d, int(checkpoint_pattern.match(d).group(1)))
               for d in all_checkpoints if checkpoint_pattern.match(d)]

last_checkpoint = max(checkpoints, key=lambda x: x[1])[0]

last_checkpoint_path = os.path.join(MODEL_DIR, last_checkpoint)
print("Last checkpoint:", last_checkpoint_path)



# Settings

tokenizer_name = "cardiffnlp/twitter-roberta-base-sentiment"

device = "cuda" if torch.cuda.is_available() else "cpu"
seed = 42



# Initializing

set_seed(seed)

model = AutoModelForSequenceClassification.from_pretrained(
    last_checkpoint_path,
    trust_remote_code=True
).to(device)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)



# Load dataset

excel = pd.read_excel(DATASET_PATH)
dataframe = pd.DataFrame(excel)

dataset = Dataset.from_pandas(dataframe)

def tokenize_function(examples):
    tokens = tokenizer(
        text = examples['Review'],
        truncation=True,
        max_length=512
    )
    return tokens

token_dataset = dataset.map(
    tokenize_function,
    batched = True
)



# Predict

training_args = TrainingArguments(
    per_device_train_batch_size = 16,
    per_device_eval_batch_size = 16,
    dataloader_num_workers = 1,
    seed = seed,
    data_seed = seed,

)

trainer = Trainer(
    model = model,
    args = training_args,
    data_collator = data_collator,
    compute_metrics = compute_metrics,
)
predictions = trainer.predict(test_dataset = token_dataset)



# (task 6: output conclusivo)

pred_labels = np.argmax(predictions.predictions, axis=1)
dataframe.insert(1, "Promotore", pred_labels)
dataframe.to_excel(OUTPUT_PATH, index=False)