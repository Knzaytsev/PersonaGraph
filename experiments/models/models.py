from models.gnn import EncoderGNNModel
from utils.graph_utils import create_dataset, FeatureExtractor

import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.nn import GAT, GraphSAGE, GCN
from torch.nn import functional as F

import torch.utils.data as Data

from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from tqdm import tqdm

import torch.nn.functional as F

import numpy as np

from transformers import (AutoTokenizer, 
                          DataCollatorWithPadding,
                          AutoModelForSequenceClassification, 
                          TrainingArguments, 
                          Trainer,
                          )
import datasets
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, classification_report
import numpy as np
import json

from sklearn.multioutput import MultiOutputClassifier

from sklearn.linear_model import LogisticRegression

class AutoPersonaClassifier():
    def __init__(self, name, type, **kwargs) -> None:
        self.name = name
        self.type = type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @classmethod
    def load_model(self, type, **kwargs):
        models_cls = {
            'encoder+GNN': EncoderGNN,
            'encoder': Encoder,
            'embeddings+GNN': EmbeddingsGNN,
            'embeddings+plain-linear': EmbeddingsPlainLinear,
        }

        return models_cls[type](type=type, **kwargs)
    
    def run_experiments(self, dataset, valid_path, epochs, batch_size, **kwargs):
        self.train(**kwargs)
        self.validate(**kwargs)
        return {
            'ground_truth': [],
            'probas': [],
        }

    def train(self, **kwargs):
        pass
    
    def validate(self, **kwargs):
        pass

class EncoderGNN(AutoPersonaClassifier):
    def __init__(self, name, type, graph_data, model_config, labels, **kwargs) -> None:
        super().__init__(name, type, **kwargs)
        self.encoder_config = model_config['encoder']
        self.gnn_config = model_config['gnn']

        self.encoder = AutoModel.from_pretrained(**self.encoder_config)
        self.gnn = AutoGNN.load_model(self.gnn_config['type'], 
                                      in_channels=self.encoder.config.hidden_size, 
                                      out_channels=len(labels),
                                      **self.gnn_config['params'])

        self.model = EncoderGNNModel(self.encoder, self.gnn, len(labels)).to(self.device)

        self.graph_data_path = graph_data

        self.labels = labels

    def update_features(self, model, x):
        model.eval()
        with torch.no_grad():
            batches = Data.DataLoader(x, 32, shuffle=False)
            all_features = []
            for batch in tqdm(batches, total=len(x) // 32 + 1):
                attention_mask = (batch != model.encoder.config.pad_token_id).to(device=self.device, dtype=torch.long)

                features = model.encoder(input_ids=batch, attention_mask=attention_mask)[0]
                features = features[:, 0, :]

                all_features.append(features.detach().cpu())
            
            encoder_features = torch.cat(all_features).cpu()
        return encoder_features

    def train(self, model, optimizer, criterion, x, y, encoder_features, idx, edge_index, edge_weight):
        optimizer.zero_grad()

        attention_mask = (x != model.encoder.config.pad_token_id).to(device=self.device, dtype=torch.long)

        logits = model(x, attention_mask, encoder_features.to(self.device), idx, edge_index, edge_weight)

        loss = criterion(logits, y[idx].to(dtype=torch.float))

        loss.backward()
        optimizer.step()

        encoder_features.detach_().cpu()

        return loss.item()

    @torch.no_grad
    def test(self, model, loader, x, y, encoder_features, edge_index, edge_attr):
        all_probas = []
        all_predictions = []
        all_true = []

        for batch_idx in loader:
            attention_mask = (x != model.encoder.config.pad_token_id).to(device=self.device, dtype=torch.long)

            logits = model(x, attention_mask, encoder_features.to('cuda'), batch_idx, edge_index, edge_attr)

            probas = F.sigmoid(logits)
            predictions = (probas > 0.35).to(torch.long).detach().cpu()

            all_probas.append(probas.detach().cpu())
            all_true.append(y[batch_idx].detach().cpu())
            all_predictions.append(predictions)

        all_predictions = torch.cat(all_predictions)
        all_probas = torch.cat(all_probas)
        all_true = torch.cat(all_true)

        return all_true, all_probas, all_predictions

    def run_experiments(self, dataset, valid_path, epochs, batch_size, **kwargs):
        train_labels = dataset['plain']
        test_labels = valid_path

        meta_dict = create_dataset('persona', train_labels, test_labels, 
                                   self.labels, self.graph_data_path, 'tokenizer', 
                                   self.encoder_config['pretrained_model_name_or_path'])

        dataset = PygNodePropPredDataset('ogbn-persona',  meta_dict=meta_dict)
        split_idx = dataset.get_idx_split()
        data = dataset[0]

        optimizer = torch.optim.AdamW(
            [
                {'params': self.model.encoder.parameters(), 'lr': 2e-5},
                {'params': self.model.classifier.parameters(), 'lr': 2e-5},
                {'params': self.model.gnn.parameters(), 'lr': 2e-3},
            ],
            lr=2e-3
        )

        criterion = torch.nn.BCEWithLogitsLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        x, y, edge_attr, edge_index = data.x.to(self.device), data.y.to(self.device), data.edge_attr.to(self.device), data.edge_index.to(self.device)
        train_idx = split_idx['train'].to(self.device)
        val_idx = split_idx['valid'].to(self.device)

        train_idx = Data.TensorDataset(train_idx)
        val_idx = Data.TensorDataset(val_idx)

        idx_loader_train = Data.DataLoader(train_idx, batch_size, shuffle=True)
        idx_loader_val = Data.DataLoader(val_idx, batch_size=batch_size)

        best_val_score = 0
        
        for epoch in range(1, epochs + 1):
            encoder_features = self.update_features(self.model, x)

            self.model.train()
            pb = tqdm(idx_loader_train, total=len(train_idx) // batch_size + 1)
            lossess = []
            for batch_idx in pb:
                loss = self.train(self.model, optimizer, criterion, x, y, encoder_features, batch_idx, edge_index, edge_attr)
                lossess.append(loss)
                pb.set_postfix({'loss': loss, 'epoch': epoch, 'mean losses': np.mean(lossess[-10:])})
            
            self.model.eval()
            true, probas, predictions = self.test(self.model, idx_loader_val, x, y, encoder_features, edge_index, edge_attr)
            val_score = f1_score(true, predictions, average='weighted')

            if val_score > best_val_score:
                best_val_score = val_score
                best_true = true
                best_probas = probas

            scheduler.step()

        torch.cuda.empty_cache()

        return {
            'ground_truth': best_true.tolist(),
            'probas': best_probas.tolist(),
        }

class AutoGNN():
    @classmethod
    def load_model(self, type, **kwargs):
        models_cls = {
            'GAT': GAT,
            'GraphSAGE': GraphSAGE,
            'GCN': GCN,
        }

        return models_cls[type](**kwargs)

class Encoder(AutoPersonaClassifier):
    def __init__(self, name, type, model_config, labels, **kwargs) -> None:
        super().__init__(name, type, **kwargs)

        self.label2id = {label: i for i, label in enumerate(labels)}
        self.id2label = {i: label for i, label in enumerate(labels)}

        self.encoder = AutoModelForSequenceClassification.from_pretrained(**model_config['encoder'], 
                                                                          num_labels=len(labels), 
                                                                          ignore_mismatched_sizes=True,
                                                                          label2id=self.label2id,
                                                                          id2label=self.id2label).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_config['encoder']['pretrained_model_name_or_path'])

        self.training_args = model_config['training_args']

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def compute_metrics(self, eval_preds):
        logits, labels = eval_preds

        if len(labels.shape) > 1 and labels.shape[-1] != 1:
            predictions = self.sigmoid(logits)
            predictions = (predictions > 0.35).astype(int).reshape(-1)
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

    def preprocess_data(self, samples, labels_map):
        if 'target' not in samples:
            inputs = self.tokenizer(samples['text'])
            if isinstance(samples['labels'][0], list):
                labels = []
                for all_labels in samples['labels']:
                    sample_labels = [0.]*len(labels_map)
                    for label in all_labels:
                        if label not in labels_map:
                            continue
                        label_id = labels_map[label]
                        sample_labels[label_id] = 1.
                    labels.append(sample_labels)
            else:
                labels = [labels_map[label] for label in samples['labels']]
        else:
            inputs = self.tokenizer(samples['text'], samples['labels'])
            labels = [labels_map[label] for label in samples['target'] if label in labels_map]
        return {
            **inputs,
            'labels': labels,
        }

    def train(self, dataset, batch_size, epochs, **kwargs):
        dataset = dataset['huggingface']

        raw_datasets = datasets.load_from_disk(dataset)

        tokenized_datasets = raw_datasets.map(
            lambda x: self.preprocess_data(x, self.label2id),
            batched=True,
        )

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        training_args = TrainingArguments(
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            **self.training_args
        )

        trainer = Trainer(
            model=self.encoder,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            tokenizer=self.tokenizer,
        )
        trainer.train()
        trainer.save_model(self.training_args['output_dir'] + '/model')

    @torch.no_grad
    def validate(self, eval_dataset, model_path, **kwargs):
        eval_dataset = [json.loads(line) for line in open(eval_dataset).readlines()]
        eval_dataset = list(filter(lambda x: 'None' not in x['labels'], eval_dataset))

        self.encoder = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)

        all_predictions = []
        all_labels = []
        all_probas = []
        for row in tqdm(eval_dataset):
            labels = [0]*len(self.label2id)
            for label in row['labels']:
                labels[self.label2id[label]] = 1

            inputs = self.tokenizer(row['text'], truncation=True, padding=True, return_tensors='pt')
            logits = self.encoder(**inputs.to(self.device)).logits
            
            probas = torch.sigmoid(logits).detach().cpu()
            # predictions = (probas > 0.35).astype(int)

            all_probas.append(probas)
            # all_predictions.append(predictions)
            all_labels.append(labels)
        
        all_probas = torch.cat(all_probas).tolist()

        torch.cuda.empty_cache()

        return all_labels, all_probas

    def run_experiments(self, dataset, valid_path, epochs, batch_size, **kwargs):
        self.train(dataset, batch_size, epochs)
        del self.encoder
        torch.cuda.empty_cache()
        true, probas = self.validate(valid_path, self.training_args['output_dir'] + '/model')

        return {
            'ground_truth': true,
            'probas': probas,
        }
    
class EmbeddingsGNN(AutoPersonaClassifier):
    def __init__(self, name, type, graph_data, model_config, labels, **kwargs) -> None:
        super().__init__(name, type, **kwargs)

        self.graph_data_path = graph_data

        self.gnn_config = model_config['gnn']

        self.feature_extractor_config = model_config['feature_extractor']

        self.labels = labels

        self.model_path = self.feature_extractor_config.get('model_path', None)

    def train(self, model, optimizer, criterion, x, y, idx, edge_index, edge_weight):
        optimizer.zero_grad()

        logits = model(x, edge_index, edge_weight)

        loss = criterion(logits[idx], y[idx].to(dtype=torch.float))

        loss.backward()
        optimizer.step()

        return loss.item()

    @torch.no_grad
    def test(self, model, idx, x, y, edge_index, edge_attr):
        logits = model(x, edge_index, edge_attr)

        probas = F.sigmoid(logits)

        return y[idx].detach().cpu(), probas[idx].detach().cpu(), (probas[idx] > 0.35).to(torch.long).detach().cpu()

    def run_experiments(self, dataset, valid_path, epochs, batch_size, **kwargs):
        train_labels = dataset['plain']
        test_labels = valid_path

        meta_dict = create_dataset('persona', train_labels, test_labels, 
                                   self.labels, self.graph_data_path, 
                                   self.feature_extractor_config['type'], 
                                   self.model_path)

        dataset = PygNodePropPredDataset('ogbn-persona',  meta_dict=meta_dict)
        split_idx = dataset.get_idx_split()
        data = dataset[0]

        self.model = AutoGNN.load_model(self.gnn_config['type'], in_channels=data.x.size(-1), out_channels=data.y.size(-1),
                                         **self.gnn_config['params']).to(self.device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-3)

        criterion = torch.nn.BCEWithLogitsLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        x, y, edge_attr, edge_index = data.x.to(self.device), data.y.to(self.device), data.edge_attr.to(self.device), data.edge_index.to(self.device)
        train_idx = split_idx['train'].to(self.device)
        val_idx = split_idx['valid'].to(self.device)

        # train_idx = Data.TensorDataset(train_idx)
        # val_idx = Data.TensorDataset(val_idx)

        best_val_score = 0
        
        pb = tqdm(range(1, epochs + 1))
        lossess = []
        for epoch in pb:
            self.model.train()
            loss = self.train(self.model, optimizer, criterion, x, y, train_idx, edge_index, edge_attr)
            lossess.append(loss)
            pb.set_postfix({'loss': loss, 'epoch': epoch, 'mean losses': np.mean(lossess[-10:])})
            
            self.model.eval()
            true, probas, predictions = self.test(self.model, val_idx, x, y, edge_index, edge_attr)
            val_score = f1_score(true, predictions, average='weighted')

            if val_score > best_val_score:
                best_val_score = val_score
                best_true = true
                best_probas = probas

            scheduler.step()

        torch.cuda.empty_cache()

        return {
            'ground_truth': best_true.tolist(),
            'probas': best_probas.tolist(),
        }

class EmbeddingsPlainLinear(AutoPersonaClassifier):
    def __init__(self, name, type, labels, model_config, **kwargs) -> None:
        super().__init__(name, type, **kwargs)

        self.feature_extractor_config = model_config['feature_extractor']

        self.labels = labels

        self.model_path = self.feature_extractor_config.get('model_path', None)

        self.feature_extractor = FeatureExtractor.load_model(self.feature_extractor_config['type'], model_path=self.model_path)
        
    def get_targets(self, samples, labels, multi=True):
        if multi:
            targets = [0]*len(labels)

            for label in samples['labels']:
                if label not in labels:
                    continue
                targets[labels.index(label)] = 1
        else:
            targets = labels.index(samples['labels'][0])

        return {
            'text': samples['text'],
            'labels': targets
        }

    def run_experiments(self, dataset, valid_path, epochs, batch_size, **kwargs):
        dataset = dataset['huggingface']
        raw_datasets = datasets.load_from_disk(dataset)
        raw_datasets = raw_datasets.map(lambda x: self.get_targets(x, self.labels))

        train = raw_datasets['train']
        test = raw_datasets['test']

        train = train.filter(lambda x: sum(x['labels']) > 0)
        test = test.filter(lambda x: sum(x['labels']) > 0)

        X_train = self.feature_extractor.extract_features(train['text'])
        y_train = train['labels']

        try:
            clf = MultiOutputClassifier(estimator=LogisticRegression(max_iter=1000)).fit(X_train, y_train)
        except:
            return {
                'ground_truth': [],
                'probas': []
            }

        eval_dataset = [json.loads(line) for line in open(valid_path).readlines()]
        eval_dataset = list(filter(lambda x: 'None' not in x['labels'], eval_dataset))

        all_labels = []
        texts = []
        for row in tqdm(eval_dataset):
            labels = [0]*len(self.labels)
            for label in row['labels']:
                labels[self.labels.index(label)] = 1
            all_labels.append(labels)
            texts.append(row['text'])

        X_test = np.array(self.feature_extractor.extract_features(texts))
        y_test = all_labels
        y_hat = clf.predict_proba(X_test)

        y_hat = np.array(y_hat)[:, :, 1].T

        return {
            'ground_truth': y_test,
            'probas': y_hat.tolist()
        }