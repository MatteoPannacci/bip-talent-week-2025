import os
import re
import pickle
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
PICKLE_PATH = os.path.join(MAIN_DIR, "matteo_pannacci_model.pickle")



# Settings

tokenizer_name = "cardiffnlp/twitter-roberta-base-sentiment"

device = "cuda" if torch.cuda.is_available() else "cpu"
seed = 42



# Initializing

set_seed(seed)

with open(PICKLE_PATH, "rb") as f:
    model = pickle.load(f)
model.to(device)

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
dataframe.insert(1, "promotore_pred", pred_labels)
dataframe.to_excel(OUTPUT_PATH, index=False)