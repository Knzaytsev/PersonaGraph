import argparse
import datasets
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--test_size', default=.2, type=float)
    args = parser.parse_args()

    if args.dataset.endswith('.jsonl'):
        data = [json.loads(line) for line in open(args.dataset).readlines()]
    else:
        raise NotImplementedError()
    
    dataset = datasets.Dataset.from_list(data)
    print(dataset)

    dataset = dataset.train_test_split(args.test_size)

    print(dataset)

    dataset.save_to_disk(args.output)

