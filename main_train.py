import os
import torch
import pandas as pd
from datasets import load_dataset, Dataset

from utilities import *

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    set_seed
)



os.environ["TOKENIZERS_PARALLELISM"] = "false"



# Constants

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(MAIN_DIR, "data/data_classification.xlsx")
LOG_DIR = os.path.join(MAIN_DIR, "log")
MODEL_DIR = os.path.join(MAIN_DIR, "model")



# Settings

model_name = "gosorio/robertaSentimentFT_TripAdvisor"
tokenizer_name = "cardiffnlp/twitter-roberta-base-sentiment"

device = "cuda" if torch.cuda.is_available() else "cpu"
seed = 42



# Initializing

set_seed(seed)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    ignore_mismatched_sizes = True,
    output_attentions = False,
    output_hidden_states = False,
    num_labels = 2,
    trust_remote_code = True,
).to(device)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)



# Load dataset

excel = pd.read_excel(DATASET_PATH)
dataframe = pd.DataFrame(excel)

dataset = Dataset.from_pandas(dataframe)
dataset = dataset.train_test_split(test_size=0.2, seed=seed)  # (task 1: split)


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



# Create Trainer

training_args = TrainingArguments(

    # hyperparameters
    num_train_epochs = 1,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size = 8,
    warmup_steps = 50,
    weight_decay = 1e-3,
    learning_rate = 1e-4,
    gradient_accumulation_steps = 40,

    # model saving
    save_strategy = "steps",
    save_steps = 50,
    output_dir = MODEL_DIR,
    overwrite_output_dir = True,
    save_total_limit = 5,
    metric_for_best_model = 'eval_accuracy',
    greater_is_better = True,

    # logging
    report_to = ["tensorboard"],
    logging_strategy = "steps",
    logging_steps = 50,
    logging_first_step = True,
    logging_dir = LOG_DIR,

    # evaluation
    eval_strategy = "steps",
    eval_steps = 50,

    # other
    dataloader_num_workers = 1,
    seed = seed,
    data_seed = seed,
    load_best_model_at_end = True,

)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = token_dataset["train"],
    eval_dataset = token_dataset["test"],
    data_collator = data_collator,
    compute_metrics = compute_metrics,
    callbacks = [EarlyStoppingCallback(5, 0)]
)



# Train (task 2: train)

trainer.train()



# (task 3: ROC)



# (task 4: confusion matrix)