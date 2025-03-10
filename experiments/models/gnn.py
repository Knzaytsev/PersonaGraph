import torch
from torch import nn
from torch.nn import functional as F

class ClassifierHead(nn.Module):
    def __init__(self, hidden_size, classifier_dropout, num_labels) -> None:
        super(ClassifierHead, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class EncoderClassification(nn.Module):
    def __init__(self, pretrained_model, num_labels) -> None:
        super(EncoderClassification, self).__init__()

        self.encoder = pretrained_model
        self.classifier = ClassifierHead(self.encoder.config.hidden_size, 
                                         self.encoder.config.classifier_dropout 
                                         if self.encoder.config.classifier_dropout is not None 
                                         else self.encoder.config.hidden_dropout_prob, 
                                         num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
        logits = self.classifier(outputs)
        return logits

class EncoderGNNModel(nn.Module):
    def __init__(self, pretrained_model, gnn_model, num_labels, m=0.7) -> None:
        super(EncoderGNNModel, self).__init__()
        self.m = m

        self.encoder = pretrained_model
        self.gnn = gnn_model

        self.classifier = ClassifierHead(self.encoder.config.hidden_size, 
                                         self.encoder.config.classifier_dropout 
                                         if self.encoder.config.classifier_dropout is not None 
                                         else self.encoder.config.hidden_dropout_prob, 
                                         num_labels)

    def forward(self, input_ids, attention_mask, encoder_features, idx, edge_index, edge_weight):
        input_ids, attention_mask = input_ids[idx], attention_mask[idx]
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
        encoder_features[idx] = outputs[:, 0, :]

        classifier_logits = self.classifier(outputs)
        graph_logits = self.gnn(encoder_features, edge_index, edge_weight=edge_weight)
        graph_logits = graph_logits[idx]


        logits = F.sigmoid(graph_logits)*self.m + F.sigmoid(classifier_logits)*(1-self.m)

        return torch.log(logits)