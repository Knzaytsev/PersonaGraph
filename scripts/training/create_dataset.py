import argparse
import datasets
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', required=True)
    parser.add_argument('--val_data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--test_size', default=.2, type=float)
    args = parser.parse_args()

    if args.train_data.endswith('.jsonl'):
        train = [json.loads(line) for line in open(args.train_data).readlines()]
    else:
        raise NotImplementedError()
    
    if args.val_data.endswith('.jsonl'):
        valid = [json.loads(line) for line in open(args.val_data).readlines()]
    else:
        raise NotImplementedError()
    
    train = datasets.Dataset.from_list(train)
    valid = datasets.Dataset.from_list(valid)

    dataset = datasets.DatasetDict()
    dataset['train'] = train
    dataset['test'] = valid

    print(dataset)

    dataset.save_to_disk(args.output)

