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

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    # true_labels = [label_names[l] for label in labels for l in label if l != -100]
    # true_predictions = [
    #     label_names[p] for prediction, label in zip(predictions, labels) 
    #     for (p, l) in zip(prediction, label) if l != -100
    # ]

    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def preprocess_data(samples, labels_map, max_len=512):
    inputs = tokenizer(samples['text'], samples['labels'])
    
    return {
        **inputs,
        'labels': [labels_map[label] for label in samples['target']],
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda', 'mps'])
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config, 'r'))
    model_args = config.pop('model_args', {})

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, **config.get('tokenizer_args', {}))

    raw_datasets = datasets.load_from_disk(args.dataset)

    tokenized_datasets = raw_datasets.map(
        lambda x: preprocess_data(x, model_args['labels_map']),
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
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