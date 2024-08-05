import datasets
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
import numpy as np
import argparse
import yaml
import torch
import torch.nn.functional as F

LABELS = [
    'Experiences',
    'Characteristics',
    'Routines or Habits',
    'Goals or Plans',
    'Relationship',
    'None'
]

class MLPLinear(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MLPLinear, self).__init__()
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x):
        return F.log_softmax(self.lin(x), dim=-1)

def train_step(model, x, y_true, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(x)
    loss = F.nll_loss(out, y_true.squeeze(1))
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test_step(model, x, y):
    model.eval()

    out = model(x)
    y_pred = out.argmax(dim=-1, keepdim=True).detach().cpu()

    train_acc = accuracy_score(**{
        'y_true': y.detach().cpu(),
        'y_pred': y_pred,
    })
    valid_acc = accuracy_score(**{
        'y_true': y.detach().cpu(),
        'y_pred': y_pred,
    })
    test_acc = accuracy_score(**{
        'y_true': y.detach().cpu(),
        'y_pred': y_pred,
    })

    return (train_acc, valid_acc, test_acc), out

def get_targets(samples, labels, multi=False):
    if multi:
        targets = [0]*len(labels)

        for label in samples['labels']:
            targets[labels.index(label)] = 1
    else:
        targets = labels.index(samples['labels'][0])

    return {
        'text': samples['text'],
        'labels': targets
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--multi', action='store_true')
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config, 'r'))

    raw_datasets = datasets.load_from_disk(args.dataset)
    raw_datasets = raw_datasets.map(lambda x: get_targets(x, LABELS, args.multi))

    train = raw_datasets['train']
    test = raw_datasets['test']

    vec = TfidfVectorizer()
    X_train = vec.fit_transform(train['text'])
    y_train = train['labels']

    if args.multi:
        clf = MultiOutputClassifier(estimator=LogisticRegression()).fit(X_train, y_train)
    else:
        clf = LogisticRegression().fit(X_train, y_train)

    X_test = vec.transform(test['text'])
    y_test = test['labels']
    y_hat = clf.predict(X_test)

    print(classification_report(y_test, y_hat, target_names=LABELS))
    print(accuracy_score(y_test, y_hat))

    device = 'mps'
    x_train = torch.Tensor(X_train.toarray())
    model = MLPLinear(x_train.size(-1), len(LABELS)).to(device)

    x_train = x_train.to(device)
    y_train_true = torch.LongTensor(y_train).unsqueeze(1).to(device)

    x_test = torch.Tensor(X_test.toarray()).to(device)
    y_test_true = torch.LongTensor(y_test).unsqueeze(1).to(device)
    
    for run in range(1):
        import gc
        gc.collect()
        print(sum(p.numel() for p in model.parameters()))
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        best_valid = 0
        best_out = None
        for epoch in range(1, 1000):
            loss = train_step(model, x_train, y_train_true, optimizer)
            result, out = test_step(model, x_test, y_test_true)
            train_acc, valid_acc, test_acc = result
            if valid_acc > best_valid:
                best_valid = valid_acc
                best_out = out.cpu().exp()
        
            print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')