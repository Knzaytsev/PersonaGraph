import argparse
import yaml
import json
import os
from utils import utils
from models.models import AutoPersonaClassifier

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))

    random_sampled_datasets = config['random_sampled_datasets']

    train = [json.loads(line) for line in open(config['train_labels']).readlines()]
    valid = [json.loads(line) for line in open(config['valid_labels']).readlines()]
    
    train = list(filter(lambda x: all(label in config['labels'] for label in x['labels']), train))
    valid = list(filter(lambda x: all(label in config['labels'] for label in x['labels']), valid))

    metrics = []
    for n_run in range(config['n_runs']):
        interim_output_path = os.path.join(random_sampled_datasets['interim_output'], f'run_{n_run}')
        processed_output_path = os.path.join(random_sampled_datasets['processed_output'], f'run_{n_run}')

        output_paths = utils.make_random_samples(train, valid, random_sampled_datasets['ratios'], 
                                                 interim_output_path, processed_output_path)

        for model_config in config['models']:
            for output_path in output_paths:
                n_epochs = model_config.get('n_epochs', config['n_epochs'])
                classifier = AutoPersonaClassifier.load_model(**model_config, labels=config['labels'])
                eval_metrics = classifier.run_experiments(output_path, config['valid_labels'], 
                                                                  n_epochs, config['batch_size'])
                metrics.append({
                    'n_run': n_run,
                    'ratio': output_path['ratio'],
                    'model': classifier.name,
                    **eval_metrics,
                })
    
    if os.path.exists(config['save_metrics_path']):
        os.remove(config['save_metrics_path'])

    with open(config['save_metrics_path'], 'a+') as f:
        for line in metrics:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
