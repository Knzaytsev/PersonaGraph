from collections import Counter
import random
import copy
import os
import json
import datasets

def make_random_samples(train, valid, ratios, interim_output, processed_output):
    labels_statistics = Counter([label for line in valid for label in line['labels']])
    labels_statistics = {label: cnt / sum(labels_statistics.values()) 
                         for label, cnt in labels_statistics.items()}
    train = [{**line, 'weight': sum(labels_statistics[label] for label in line['labels'])} for line in train]

    if not os.path.exists(interim_output):
        os.makedirs(interim_output)

    output_paths = []

    for ratio in ratios:
        n_samples = int(len(train)*ratio)

        ratio_train = random.choices(train, weights=[line['weight'] for line in train], k=n_samples)
        ratio_valid = copy.deepcopy(valid)

        file_name = f'train_{ratio}.jsonl'
        file_path = os.path.join(interim_output, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)

        with open(file_path, 'a+') as f:
            for line in ratio_train:
                f.write(json.dumps(line, ensure_ascii=False) + '\n')

        ratio_train = datasets.Dataset.from_list(ratio_train)
        ratio_valid = datasets.Dataset.from_list(ratio_valid)

        dataset = datasets.DatasetDict()
        dataset['train'] = ratio_train
        dataset['test'] = ratio_valid

        dataset_folder = os.path.join(processed_output, f'sample-{str(ratio)}')
        dataset.save_to_disk(dataset_folder)

        output_paths.append({
            'ratio': ratio,
            'plain': file_path,
            'huggingface': dataset_folder,
        })
    return output_paths

def gen_batches(x, batch_size):
    for i in range(0, len(x) // batch_size + 1):
        yield x[i*batch_size:(i+1)*batch_size]