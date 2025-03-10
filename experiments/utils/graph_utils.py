import torch
import numpy as np
import json

from ogb.io import DatasetSaver
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import networkx as nx

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

from transformers import BatchEncoding

import os

from utils.utils import gen_batches

def average_pool(last_hidden_states,
                 attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

class FeatureExtractor():
    def extract_features(self, data):
        raise NotImplementedError()

    @classmethod
    def load_model(self, model_name, **kwargs):
        if model_name in ['count-vectorizer', 'tf-idf-vectorizer']:
            return BOWExtractor(model_name)
        elif model_name == 'sentence-transformers':
            return SentenceTransformerExtractor(**kwargs)
        elif model_name == 'tokenizer':
            return TokenizerExtractor(**kwargs)
        elif model_name == 'transformer':
            return TransformerExtractor(**kwargs)
        else:
            raise NotImplementedError()

class BOWExtractor(FeatureExtractor):
    def __init__(self, extractor, **kwargs) -> None:
        if extractor == 'count-vectorizer':
            self.vec = CountVectorizer()
        else:
            self.vec = TfidfVectorizer()
        self.fitted = False

    def extract_features(self, data):
        if not self.fitted:
            self.fitted = True
            features = self.vec.fit_transform(data)
        else:
            features = self.vec.transform(data)
        return features.toarray().astype(np.float32)

class SentenceTransformerExtractor(FeatureExtractor):
    def __init__(self, model_path, **kwargs) -> None:
        self.model = SentenceTransformer(model_path, device='cuda')
        self.prefix = 'query: ' if 'e5' in model_path else ''

    def extract_features(self, data):
        embeddings = self.model.encode(data, prompt=self.prefix, batch_size=8, show_progress_bar=True)

        return embeddings

class TransformerExtractor(FeatureExtractor):
    def __init__(self, model_path, device='cuda', **kwargs) -> None:
        self.model = AutoModel.from_pretrained(model_path, add_pooling_layer=False).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def extract_features(self, data):
        batches = gen_batches(data, 4)
        outputs = []
        for batch in batches:
            if not batch:
                continue
            input_ids = self.tokenizer(batch, truncation=True, padding='max_length', max_length=256, return_tensors='pt')
            out = self.model(**input_ids.to(self.model.device)).last_hidden_state
            outputs.append(out.detach().cpu())
        outputs = torch.cat(outputs)
        return outputs[:, 0, :].numpy()

class TokenizerExtractor(FeatureExtractor):
    def __init__(self, model_path, **kwargs) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def extract_features(self, data):
        input_ids = self.tokenizer(data, truncation=True, padding='max_length', max_length=256, return_tensors='np')
        return input_ids['input_ids']

def create_dataset(dataset, train_labels, test_labels, label_names, graph, feature_extractor, model_path, output='persona-ogbn'):
    saver = DatasetSaver(dataset_name = 'ogbn-' + dataset, is_hetero = False, version = 1)

    train_labels = [json.loads(line) for line in open(train_labels).readlines()]
    valid_labels = [json.loads(line) for line in open(test_labels).readlines()]

    train_labels = {label['text']: {'labels': label['labels'], 'split': 'train'} for label in train_labels}
    test_labels = {label['text']: {'labels': label['labels'], 'split': 'test'} for label in valid_labels}
    valid_labels = {label['text']: {'labels': label['labels'], 'split': 'valid'} for label in valid_labels}

    G: nx.Graph = nx.gexf.read_gexf(graph)

    label2id = {label: i for i, label in enumerate(label_names)}

    subgraph = nx.subgraph_view(G, filter_node=lambda x: x in train_labels or x in valid_labels or x in test_labels)

    # labels = {
    #     **{k: '; '.join(v['labels']) for k, v in train_labels.items()}, 
    #     **{k: '; '.join(v['labels']) for k, v in test_labels.items()}, 
    #     **{k: '; '.join(v['labels']) for k, v in valid_labels.items()}
    #     }
    # print(labels)
    # nx.set_node_attributes(subgraph, labels, "labels")
    # nx.gexf.write_gexf(subgraph, "big_weighted_label_graph.gexf")
    # raise ValueError()

    text2id = {text: i for i, text in enumerate(subgraph.nodes)}
    id2text = {i: text for text, i in text2id.items()}

    subgraph = nx.relabel.relabel_nodes(subgraph, text2id)

    feature_extractor = FeatureExtractor.load_model(feature_extractor, model_path=model_path)

    graph_list = []
    labels_list = []
    train_ids = []
    test_ids = []
    valid_ids = []
    for g, labels in [(subgraph, {**train_labels, **test_labels, **valid_labels})]:
        graph = dict()
        edges = list(g.edges)
        edges += [(edge[1], edge[0]) for edge in g.edges]
        weighted_edges = []
        weighted_edges_ = nx.get_edge_attributes(g, "weight")
        for edge in edges:
            if edge in weighted_edges_:
                weighted_edges.append(weighted_edges_[edge])
            else:
                weighted_edges.append(weighted_edges_[(edge[1], edge[0])])
        graph['edge_index'] = np.array(edges).transpose()
        graph['num_nodes'] = len(g.nodes)
        graph['edge_feat'] = np.array(weighted_edges).reshape(-1, 1)

        features = feature_extractor.extract_features([id2text[node] for node in g.nodes])
        if isinstance(features, dict) or isinstance(features, BatchEncoding):
            for k, v in features.items():
                graph['node_' + k] = v
        else:
            graph['node_feat'] = features

        for node in g.nodes:
            if labels[id2text[node]]['split'] == 'train':
                train_ids.append(node)
            elif labels[id2text[node]]['split'] in ['test', 'valid']:
                test_ids.append(node)
                valid_ids.append(node)
        
            new_labels = [0.]*len(label_names)
            for l in labels[id2text[node]]['labels']:
                if l not in label_names:
                    continue
                new_labels[label2id[l]] = 1.
            labels_list.append(new_labels)
            
        graph_list.append(graph)

    saver.save_graph_list(graph_list)

    saver.save_target_labels(np.array(labels_list))

    assert len(train_ids) > 0 and len(test_ids) > 0 and len(valid_ids) > 0, (len(train_ids), len(test_ids), len(valid_ids))

    split_idx = {'train': torch.LongTensor(train_ids), 'test': torch.LongTensor(test_ids), 'valid': torch.LongTensor(valid_ids)}

    saver.save_split(split_idx, 'random')

    # prepare mapping information first and store it under this directory (empty below).
    os.makedirs(output, exist_ok=True)
    with open(os.path.join(output, 'README.md'), 'w') as f:
        f.write('')

    saver.copy_mapping_dir(output)

    saver.save_task_info(task_type = 'classification', eval_metric = 'acc', num_classes = len(label_names))

    meta_dict = saver.get_meta_dict()

    return meta_dict

