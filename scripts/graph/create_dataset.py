import argparse
import networkx as nx
import json
from ogb.io import DatasetSaver
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
import random
import os

LABELS = [
    'Experiences',
    'Characteristics',
    'Routines or Habits',
    'Goals or Plans',
    'Relationship',
    'None'
]

def feature_extractor(data, extractor, model_path=None):
    if extractor in ['count-vectorizer', 'tf-idf-vectorizer']:
        if extractor == 'count-vectorizer':
            vec = CountVectorizer()
        else:
            vec = TfidfVectorizer()
        
        features = vec.fit_transform(data)

        return features.toarray()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', required=True)
    parser.add_argument('--train_labels', required=True)
    parser.add_argument('--test_labels', required=True)
    parser.add_argument('--dataset_name', default='persona')
    parser.add_argument('--feature_extractor', 
                        choices=['transformers', 'count-vectorizer', 'tf-idf-vectorizer'], 
                        default='tf-idf-vectorizer')
    parser.add_argument('--model_path')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()


    saver = DatasetSaver(dataset_name = 'ogbn-' + args.dataset_name, is_hetero = False, version = 1)

    train_labels = [json.loads(line) for line in open(args.train_labels).readlines()]
    train_labels = {label['text']: {'labels': label['labels'][0], 'split': 'train'} for label in train_labels}

    valid_labels = [json.loads(line) for line in open(args.test_labels).readlines()]

    random.shuffle(valid_labels)
    test_labels = valid_labels[:int(0.5)]
    valid_labels = valid_labels[int(0.5):]

    test_labels = {label['text']: {'labels': label['labels'][0], 'split': 'test'} for label in test_labels}
    valid_labels = {label['text']: {'labels': label['labels'][0], 'split': 'valid'} for label in valid_labels}

    G: nx.Graph = nx.gexf.read_gexf(args.graph)

    label2id = {label: i for i, label in enumerate(LABELS)}
    text2id = {text: i for i, text in enumerate(set(list(train_labels.keys()) + list(valid_labels.keys())))}
    id2text = {i: text for text, i in text2id.items()}

    subgraph = nx.subgraph_view(G, filter_node=lambda x: x in train_labels or x in valid_labels)
    subgraph = nx.relabel.relabel_nodes(subgraph, text2id)


    graph_list = []
    labels_list = []
    train_ids = []
    test_ids = []
    valid_ids = []
    for g, labels in [(subgraph, {**train_labels, **test_labels, **valid_labels})]:
        graph = dict()
        graph['edge_index'] = np.array(list(g.edges) + [(edge[1], edge[0]) for edge in g.edges]).transpose()
        graph['num_nodes'] = len(g.nodes)
        graph['node_feat'] = feature_extractor([id2text[node] for node in g.nodes], args.feature_extractor)
        # graph['node_text'] = [id2text[node] for node in g.nodes]
        for node in g.nodes:
            if labels[id2text[node]]['split'] == 'train':
                train_ids.append(node)
            elif labels[id2text[node]]['split'] == 'test':
                test_ids.append(node)
            else:
                valid_ids.append(node)
        labels = [label2id[labels[id2text[node]]['labels']] for node in g.nodes]
        graph_list.append(graph)
        labels_list.append(labels)

    saver.save_graph_list(graph_list)

    saver.save_target_labels(np.array(labels_list).reshape((-1, 1)))

    split_idx = {'train': train_ids, 'test': test_ids, 'valid': valid_ids}

    saver.save_split(split_idx, 'random')

    # prepare mapping information first and store it under this directory (empty below).
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, 'README.md'), 'w') as f:
        f.write('')

    saver.copy_mapping_dir(args.output)

    saver.save_task_info(task_type = 'classification', eval_metric = 'acc', num_classes = len(LABELS))

    meta_dict = saver.get_meta_dict()

    with open(os.path.join(args.output, 'meta_dict.json'), 'w') as f:
        json.dump(meta_dict, f)

    saver.zip()
    saver.cleanup()