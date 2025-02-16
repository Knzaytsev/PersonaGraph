from transformers import (AutoTokenizer, 
                          AutoModelForSequenceClassification, )
import datasets
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import numpy as np
import argparse
import yaml
import torch
import json
from tqdm import tqdm

LABELS = [
    'Experiences',
    'Characteristics',
    'Routines or Habits',
    'Goals or Plans',
    'Relationship',
    'None'
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_path', required=True, nargs='+')
    parser.add_argument('--eval_dataset', required=True)
    parser.add_argument('--labels', nargs='+', default=LABELS)
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda', 'mps'])
    args = parser.parse_args()

    eval_dataset = [json.loads(line) for line in open(args.eval_dataset).readlines()]

    for path in args.models_path:
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path).to(args.device)

        task = 'nli' if 'entailment' in model.config.label2id else 'classification'

        with torch.no_grad():
            all_predictions = []
            all_labels = []
            for row in tqdm(eval_dataset, desc=path):
                labels = [0]*len(args.labels)
                for label in row['labels']:
                    labels[args.labels.index(label)] = 1

                if task == 'nli':                    
                    inputs = tokenizer([row['text']]*len(args.labels), args.labels, 
                                       truncation=True, padding=True, return_tensors='pt')
                    logits = model(**inputs.to(model.device)).logits[:, 0]
                else:
                    inputs = tokenizer(row['text'], truncation=True, padding=True, return_tensors='pt')
                    logits = model(**inputs.to(model.device)).logits[0]
                    
                probas = torch.sigmoid(logits).detach().cpu().numpy()
                predictions = (probas > 0.5).astype(int).tolist()


                all_predictions.append(predictions)
                all_labels.append(labels)

            print(path)
            print(classification_report(all_labels, all_predictions, target_names=args.labels))
            print('-'*10)