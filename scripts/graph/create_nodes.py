import json
import os
import argparse
import networkx
from src.persona_graph import PersonaNode
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import torch
import pandas as pd
import itertools
from tqdm import tqdm

def read_raw(folders, file_name):
    raw = []
    for folder in folders:
        path = os.path.join(folder, file_name)
        if os.path.exists(path):
            file = [{**json.loads(line), 'session_id': re.search(r'(session_\d)', path).group()} 
                    for line in open(path).readlines()]
            raw.extend(file)
    return raw

def get_nodes(dialogues):
    nodes = []
    for dialogue in dialogues:
        for i, turn in enumerate(dialogue['dialog']):
            persona_list = turn['agg_persona_list']
            for j, persona in enumerate(persona_list):
                p = PersonaNode(persona, turn['id'], 
                                turn['convai2_id'], 
                                session_id=dialogue['session_id'],
                                turn=i, place=j, 
                                split=turn['convai2_id'].split(':')[0])
                p_id = p.build_id()
                nodes.append((p_id, p.__dict__))

    return nodes

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-folder', required=True)
    parser.add_argument('--prefix', default='')
    parser.add_argument('--save_path', required=True)
    args = parser.parse_args()

    dataset_folder = args.dataset_folder
    prefix = args.prefix

    sessions = [os.path.join(dataset_folder, 'session_' + str(i)) for i in range(1, 5)]

    train_raw = read_raw(sessions, f'train{prefix}.txt')
    test_raw = read_raw(sessions, f'test{prefix}.txt')
    valid_raw = read_raw(sessions, f'valid{prefix}.txt')
    
    G = networkx.Graph()
    for raw in [train_raw,]:
        G.add_nodes_from(get_nodes(raw))
    
    data = networkx.node_link.node_link_data(G)

    with open(args.save_path, 'w') as f:
        json.dump(data, f)