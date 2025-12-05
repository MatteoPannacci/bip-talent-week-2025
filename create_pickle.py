import os
import re
import pickle
import torch

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed
)



# Constants

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(MAIN_DIR, "model")
PICKLE_PATH = os.path.join(MAIN_DIR, "matteo_pannacci_model.pickle")



# Find last checkpoint

all_checkpoints = [d for d in os.listdir(MODEL_DIR) if os.path.isdir(os.path.join(MODEL_DIR, d))]

checkpoint_pattern = re.compile(r"checkpoint-(\d+)")
checkpoints = [(d, int(checkpoint_pattern.match(d).group(1)))
               for d in all_checkpoints if checkpoint_pattern.match(d)]

last_checkpoint = max(checkpoints, key=lambda x: x[1])[0]

last_checkpoint_path = os.path.join(MODEL_DIR, last_checkpoint)
print("Last checkpoint:", last_checkpoint_path)



# Load model

model = AutoModelForSequenceClassification.from_pretrained(
    last_checkpoint_path,
    trust_remote_code=True
)



# Dump model

with open(PICKLE_PATH, "wb") as f:
    pickle.dump(model, f)