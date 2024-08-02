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

def sigmoid(x):
   return 1/(1 + np.exp(-x))

def compute_metrics(eval_preds):
    logits, labels = eval_preds

    if labels.shape[-1] != 1:
        predictions = sigmoid(logits)
        predictions = (predictions > 0.5).astype(int).reshape(-1)
        labels = labels.astype(int).reshape(-1)
    else:
        predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def preprocess_data(samples, labels_map):
    if 'target' not in samples:
        inputs = tokenizer(samples['text'])
        if isinstance(samples['labels'][0], list):
            labels = []
            for all_labels in samples['labels']:
                sample_labels = [0. for i in range(len(labels_map))]
                for label in all_labels:
                    label_id = labels_map[label]
                    sample_labels[label_id] = 1.
                labels.append(sample_labels)
        else:
            labels = [labels_map[label] for label in samples['labels']]
    else:
        inputs = tokenizer(samples['text'], samples['labels'])
        labels = [labels_map[label] for label in samples['target']]
    return {
        **inputs,
        'labels': labels,
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--task', choices=['classification', 'nli'], default='nli')
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config, 'r'))
    model_args = config.pop('model_args', {})

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, **config.get('tokenizer_args', {}))

    raw_datasets = datasets.load_from_disk(args.dataset)

    labels = []
    model_config = {}
    if args.task == 'classification':
        labels = model_args['labels']

        label2id = {label: i for i, label in enumerate(labels)}
        id2label = {i: label for i, label in enumerate(labels)}

        model_config = {'label2id': label2id, 'id2label': id2label}

    if args.task == 'nli':
        labels_map = model_args['labels_map']
    else:
        labels_map = label2id

    tokenized_datasets = raw_datasets.map(
        lambda x: preprocess_data(x, labels_map),
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        ignore_mismatched_sizes=True,
        **model_config,
    ).to(args.device)

    training_args = TrainingArguments(
        args.save_path,
        **config['training_args']
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        compute_metrics=lambda x: compute_metrics(x),
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.save_path + '/model')