import argparse
import networkx
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', required=True)
    parser.add_argument('--train_labels', required=True)
    parser.add_argument('--test_labels', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    G: networkx.Graph = networkx.gexf.read_gexf(args.graph)

    train_labels = [json.loads(line) for line in open(args.train_labels).readlines()]
    test_labels = [json.loads(line) for line in open(args.test_labels).readlines()]

    labels = {label['text']: {'train_labels': label['labels'][0]} for label in train_labels}
    labels = {**labels, **{label['text']: {'test_labels': label['labels'][0]} for label in test_labels}}

    networkx.set_node_attributes(G, labels)

    networkx.gexf.write_gexf(G, args.output)