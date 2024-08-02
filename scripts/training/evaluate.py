from transformers import (AutoTokenizer, 
                          DataCollatorWithPadding,
                          AutoModelForSequenceClassification, 
                          TrainingArguments, 
                          Trainer)
import datasets
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import argparse
import yaml
import torch
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_path', required=True, nargs='+')
    parser.add_argument('--eval_dataset', required=True)
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda', 'mps'])
    args = parser.parse_args()

    eval_dataset = [json.loads(line) for line in open(args.eval_dataset).readlines()]

    for path in args.models_path:
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path).to(args.device)

        for row in eval_dataset:
            pass