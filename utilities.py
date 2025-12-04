import evaluate
import numpy as np


accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")


def compute_metrics(eval_pred):

    preds = np.argmax(eval_pred.predictions, axis=-1)
    labels = eval_pred.label_ids

    accuracy = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    precision = precision_metric.compute(predictions=preds, references=labels, zero_division=0, average='macro')["precision"]
    recall = recall_metric.compute(predictions=preds, references=labels, zero_division=0, average='macro')["recall"]
    f1 = f1_metric.compute(predictions=preds, references=labels, average='macro')["f1"]

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    return metrics