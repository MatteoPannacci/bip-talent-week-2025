import os
import re
import torch
import pickle
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
DATASET_PATH = os.path.join(MAIN_DIR, "data/data_classification.xlsx")
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
dataset = dataset.train_test_split(test_size=0.2, seed=seed)

def tokenize_function(examples):
    tokens = tokenizer(
        text = examples['Review'],
        truncation=True,
        max_length=512
    )
    tokens['label'] = examples['Promotore']
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

train_predictions = trainer.predict(test_dataset = token_dataset['train'])
test_predictions = trainer.predict(test_dataset = token_dataset['test'])

train_pred_scores = train_predictions.predictions[:, 1] - train_predictions.predictions[:, 0]
test_pred_scores  = test_predictions.predictions[:, 1] - test_predictions.predictions[:, 0]

train_pred_labels = np.argmax(train_predictions.predictions, axis=1)
test_pred_labels = np.argmax(test_predictions.predictions, axis=1)

train_ground_truths = token_dataset["train"]["Promotore"]
test_ground_truths = token_dataset["test"]["Promotore"]


# (task 3: ROC)

compute_roc(train_ground_truths, train_pred_scores, "train_ROC")
compute_roc(test_ground_truths, test_pred_scores, "test_ROC")



# (task 4: confusion matrix)

compute_confusion_matrix(train_ground_truths, train_pred_labels, "train_confusion_matrix")
compute_confusion_matrix(test_ground_truths, test_pred_labels, "test_confusion_matrix")