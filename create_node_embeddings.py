import json
import argparse
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import numpy as np

def gen_batch(x, batch_size=64):
    for i in range(0, len(x), batch_size):
        yield x[i:i+batch_size]

def get_node_embeddings(nodes, batch_size=24):
    def average_pool(last_hidden_states, attention_mask):
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device('cpu')

    model_name = "intfloat/multilingual-e5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    node_embeddings = []
    batches = gen_batch(nodes, batch_size=batch_size)
    for batch in tqdm(batches, total=len(nodes) // batch_size + 1):
        input = tokenizer(list(map(lambda x: 'query: ' + x, batch)), 
                          truncation=True, return_tensors="pt", 
                          max_length=256, padding=True)
        output = model(**input.to(device))
        embeddings = average_pool(output.last_hidden_state, input['attention_mask'])
        node_embeddings.append(embeddings.detach().cpu())
    return torch.cat(node_embeddings).numpy()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes_path', required=True)
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--batch_size', default=24, type=int)
    args = parser.parse_args()

    with open(args.nodes_path, 'r') as f:
        data = json.load(f)

    nodes = data['nodes']
    text_nodes = [node['text'] for node in nodes]

    node_embeddings = get_node_embeddings(text_nodes, args.batch_size)
    np.save(args.save_path, node_embeddings)