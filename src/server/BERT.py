import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np
import gluonnlp as nlp

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 batch_size,
                 hidden_size = 768,
                 kernel_size = 3,
                 num_classes=2,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.batch_size = batch_size
        self.dr_rate = dr_rate
        self.conv1 = nn.Sequential(
            nn.Conv1d(hidden_size, 100, 2),
            nn.GELU(),
            nn.Dropout(),
            nn.MaxPool1d(kernel_size = 4)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(hidden_size, 100, 2),
            nn.GELU(),
            nn.Dropout(),
            nn.MaxPool1d(kernel_size = 4)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(hidden_size, 100, kernel_size),
            nn.GELU(),
            nn.Dropout(),
            nn.MaxPool1d(kernel_size = 4)
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(hidden_size, 100, kernel_size),
            nn.GELU(),
            nn.Dropout(),
            nn.MaxPool1d(kernel_size = 4)
        )

        self.conv5 = nn.Sequential(
            nn.Conv1d(hidden_size, 100, 4),
            nn.GELU(),
            nn.Dropout(),
            nn.MaxPool1d(kernel_size = 4)
        )

        self.conv6 = nn.Sequential(
            nn.Conv1d(hidden_size, 100, 4),
            nn.GELU(),
            nn.Dropout(),
            nn.MaxPool1d(kernel_size = 4)
        )
        self.LSTM = nn.LSTM(600,90,batch_first=True,bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(90*2, num_classes),
            nn.Softmax(dim=1)
        )

        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        _ = _.transpose(1, 2)

        out1 = self.conv1(_)
        out1 = out1.transpose(1, 2)
        out2 = self.conv2(_)
        out2 = out2.transpose(1, 2)
        out3 = self.conv1(_)
        out3 = out3.transpose(1, 2)
        
        out4 = self.conv1(_)
        out4 = out4.transpose(1, 2)
        out5 = self.conv1(_)
        out5 = out5.transpose(1, 2)
        out6 = self.conv1(_)
        out6 = out6.transpose(1, 2)
        final_out = torch.cat([out1,out2,out3,out4,out5,out6],dim=2)

        lstm_out, (h,c) = self.LSTM(final_out)
        hidden = torch.cat((lstm_out[:,-1, :90],lstm_out[:,0, 90:]),dim=-1)
        linear_output = self.classifier(hidden.view(-1,90*2))

        return linear_output

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))
