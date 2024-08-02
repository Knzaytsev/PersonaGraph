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

def dataset_maker(raw, context=2):
    dataset = []
    for line in raw:
        dialog = line['dialog']
        for i, turn in enumerate(dialog):
            persona = "none"
            if 'persona_text' in turn:
                persona = turn['tr_persona_text']
            history = '\n'.join([dialog[i]['id'] + ': ' + dialog[i]['tr_text'] for i in range(max(0, i-context), i+1)])
            dataset.append({'history': history, 'persona': persona})
    return dataset

def save_jsonl(jsonl_object, file):
    if os.path.exists(file):
        raise ValueError(file + ' already exists!')
    with open(file, 'a+') as f:
        for line in jsonl_object:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')

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

def gen_batch(x, batch_size=64):
    for i in range(0, len(x), batch_size):
        yield x[i:i+batch_size]

def match_nodes(nodes, batch_size=24):
    def average_pool(last_hidden_states, attention_mask):
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    device = torch.device("mps")

    model_name = "intfloat/multilingual-e5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    node_embeddings = []
    batches = gen_batch(nodes, batch_size=batch_size)
    for batch in tqdm(batches, total=len(nodes) // batch_size + 1):
        input = tokenizer(list(map(lambda x: 'query: ' + x, batch)), 
                          truncation=True, return_tensors="pt", 
                          max_length=256, padding=True)
        output = model(**input.to(device))  # device = "cuda:0" or "cpu"
        embeddings = average_pool(output.last_hidden_state, input['attention_mask'])
        node_embeddings.append(embeddings)
    return torch.cat(node_embeddings).detach().cpu().numpy()

def consequent_eges(nodes):
    edges = []
    prev_idx = None
    placed_nodes = []
    for idx, node in nodes.iterrows():
        if node['is_placed']:
            placed_nodes.append(node['id'])
            continue

        if placed_nodes:
            placed_nodes.append(node['id'])
            comb_edges = itertools.combinations(placed_nodes, 2)
            edges.extend(comb_edges)
        elif prev_idx is not None:
            edges.append((prev_idx, node['id']))

        prev_idx = node['id']
        placed_nodes = []
    return edges

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-folder', required=True)
    parser.add_argument('--prefix', default='')
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
    
    # print(len(G.nodes))
    # print(train_raw[0])

    # G = networkx.Graph()
    # p1 = PersonaNode('hello', 'bye', 'aa', 0)
    # p2 = PersonaNode('hhh', 'aa', 'cc', 1)
    # G.add_nodes_from([(100, p1.__dict__)])
    # G.add_nodes_from([(101, p2.__dict__)])
    # G.add_edge(100, 101)
    data = networkx.node_link.node_link_data(G)
    nodes = data['nodes']
    nodes = pd.DataFrame(nodes)
    # nodes = nodes.set_index('id')
    nodes = nodes.sort_values(['convai2_id', 'session_id', 'turn', 'place'])
    nodes['is_placed'] = nodes.groupby(['convai2_id', 'session_id', 'bot_id', 'turn'])['place'].transform('nunique') > 1
    edges = nodes.groupby('convai2_id').apply(consequent_eges).tolist()
    edges = [edge for edges in edges for edge in edges]
    nodes['dialogue_nodes'] = nodes.groupby('convai2_id')['id'].transform(lambda x: [x.tolist()]*len(x))
    nodes = nodes.sample(100)
    node_embeddings = match_nodes(nodes['text'].tolist())
    print(node_embeddings.shape)