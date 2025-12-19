import evaluate
import numpy as np
import Levenshtein

rouge = evaluate.load("rouge")

def compute_metrics(preds, labels):
    rouge_scores = rouge.compute(predictions=preds, references=labels)

    edit_distances = [
        Levenshtein.distance(ref, pred) / max(len(ref), 1)
        for ref, pred in zip(labels, preds)
    ]

    return {
        "rouge1": rouge_scores["rouge1"],
        "rouge2": rouge_scores["rouge2"],
        "rougeL": rouge_scores["rougeL"],
        "norm_edit_distance": np.mean(edit_distances)
    }
