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

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device('cpu')

def read_raw(folders, file_name):
    raw = []
    for folder in folders:
        path = os.path.join(folder, file_name)
        if os.path.exists(path):
            file = [{**json.loads(line), 'session_id': re.search(r'(session_\d)', path).group()} 
                    for line in open(path).readlines()]
            raw.extend(file)
    return raw

def gen_batch(x, batch_size=64):
    for i in range(0, len(x), batch_size):
        yield x[i:i+batch_size]

def equal_edges(nodes):
    edges = []
    cache = dict()
    for _, node in nodes.iterrows():
        if node['id'] not in cache:
            cache[node['id']] = []
        eq_nodes = nodes.loc[nodes['convai2_id'] != node['convai2_id'], 'id'].tolist()
        for eq_node in eq_nodes:
            if eq_node not in cache:
                cache[eq_node] = []
            
            if node['id'] not in cache[eq_node] and eq_node not in cache[node['id']]:
                cache[eq_node].append(node['id'])
                cache[node['id']].append(eq_node)
                edges.append((node['id'], eq_node))
    return edges

def similar_edges(cluster, model, tokenizer, entail_idx, threshold=0.8, batch_size=24):
    combinations = []
    cache = dict()
    for row in cluster.to_dict('records'):
        if row['text'] not in cache:
            cache[row['text']] = []
        nodes = cluster[cluster['convai2_id'] != row['convai2_id']].to_dict('records')
        for node in nodes:
            if node['text'] not in cache:
                cache[node['text']] = []
            
            if row['text'] not in cache[node['text']] and node['text'] not in cache[row['text']]:
                cache[row['text']].append(node['text'])
                cache[node['text']].append(row['text'])
                combinations.append((row, node))
    
    edges = []
    batches = gen_batch(combinations, batch_size=batch_size,)
    for batch in batches:
        input = tokenizer([from_['text'] for from_, _ in batch],
                          [to_['text'] for _, to_ in batch], 
                          truncation=True, return_tensors="pt", 
                          max_length=512, padding=True)
        output = model(**input.to(device))

        probas = torch.softmax(output["logits"], -1)[:, entail_idx]
        for i, proba in enumerate(probas):
            if proba > threshold:
                left, right = batch[i]
                edges.append((left['text'], right['text']))

    return edges

def consequent_edges(nodes):
    edges = []
    prev_idx = None
    placed_nodes = []
    i = 0
    prev_turn = None
    for idx, node in nodes.iterrows():
        # if node['is_placed']:
        #     placed_nodes.append(node['text'])

        #     if prev_turn != node['turn']:
        #         i += 1
            
        #     prev_turn = node['turn']

        # if placed_nodes and i == 1:
        #     placed_nodes.append(node['text'])
        #     comb_edges = list(itertools.combinations(placed_nodes, 2))
        #     edges.extend(comb_edges)
        #     i = 0
        # elif not placed_nodes:
        #     edges.append((prev_idx, node['text']))
        # else:
        #     continue
        if prev_idx is not None:
            edges.append((prev_idx, node['text']))

        prev_idx = node['text']
        placed_nodes = []
    return edges

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes', required=True)
    parser.add_argument('--cluster_nodes', required=True)
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--entail_idx', default=0, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--save_path')
    args = parser.parse_args()

    cluster_nodes = pd.read_csv(args.cluster_nodes)
    
    data = json.load(open(args.nodes))
    list_nodes = data['nodes']
    nodes = pd.DataFrame(list_nodes)
    # nodes = nodes.set_index('id')
    nodes = nodes.sort_values(['convai2_id', 'session_id', 'turn', 'place'])
    nodes['is_placed'] = nodes.groupby(['convai2_id', 'session_id', 'bot_id', 'turn'])['place'].transform('nunique') > 1

    tqdm.pandas(desc='consequent edges')
    cons_edges = nodes.groupby('convai2_id').progress_apply(consequent_edges).tolist()
    cons_edges = [edge for edges in cons_edges for edge in edges]
    print(len(cons_edges))
    
    # tqdm.pandas(desc='equal edges')
    # eq_edges = nodes.groupby('text').progress_apply(equal_edges).tolist()
    # eq_edges = [edge for edges in eq_edges for edge in edges]
    # print(len(eq_edges))

    clusters_mapping = {row['id']: row['clusters'] for _, row in cluster_nodes.iterrows()}
    nodes['clusters'] = nodes['id'].map(clusters_mapping)

    tqdm.pandas(desc='cluster edges')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name).to(device)
    sim_edges = nodes[nodes['clusters'].notna()].groupby('clusters')
    sim_edges = sim_edges.progress_apply(lambda x: similar_edges(x, model, tokenizer, args.entail_idx, batch_size=args.batch_size)).tolist()
    sim_edges = [edge for edges in sim_edges for edge in edges]

    edges = [*cons_edges, *sim_edges]
    edges = pd.Series(['<|_|>'.join(sorted(edge)) for edge in edges]).drop_duplicates()
    edges = edges.str.split('<|_|>', regex=False).apply(tuple)

    G = networkx.Graph()
    G.add_nodes_from((node) for node in set(node['text'] for node in list_nodes))
    G.add_edges_from(edges)
    print(len(G.edges))
    print(len(G.nodes))

    networkx.gexf.write_gexf(G, args.save_path)

    # data = networkx.node_link.node_link_data(G)
    # with open(args.save_path, 'w') as f:
    #     json.dump(data, f)