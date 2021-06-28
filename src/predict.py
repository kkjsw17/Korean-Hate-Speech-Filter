import torch
from torch import nn
from torch.utils.data import Dataset
import gluonnlp as nlp
import numpy as np
from kobert.pytorch_kobert import get_pytorch_kobert_model
import sys
import os
import requests
import hashlib
from zipfile import ZipFile
from transformers import BertModel

tokenizer = {
    'url':
    'https://kobert.blob.core.windows.net/models/kobert/tokenizer/kobert_news_wiki_ko_cased-ae5711deb3.spiece',
    'fname': 'kobert_news_wiki_ko_cased-1087f8699e.spiece',
    'chksum': 'ae5711deb3'
}

pytorch_kobert = {
    'url':
    'https://kobert.blob.core.windows.net/models/kobert/pytorch/kobert_v1.zip',
    'fname': 'kobert_v1.zip',
    'chksum': '411b242919'  # 411b2429199bc04558576acdcac6d498
}

def download(url, filename, chksum, cachedir='~/kobert/'):
    f_cachedir = os.path.expanduser(cachedir)
    os.makedirs(f_cachedir, exist_ok=True)
    file_path = os.path.join(f_cachedir, filename)
    if os.path.isfile(file_path):
        if hashlib.md5(open(file_path,
                            'rb').read()).hexdigest()[:10] == chksum:
            #print('using cached model')
            return file_path
    with open(file_path, 'wb') as f:
        response = requests.get(url, stream=True)
        total = response.headers.get('content-length')

        if total is None:
            f.write(response.content)
        else:
            downloaded = 0
            total = int(total)
            for data in response.iter_content(
                    chunk_size=max(int(total / 1000), 1024 * 1024)):
                downloaded += len(data)
                f.write(data)
                done = int(50 * downloaded / total)
                sys.stdout.write('\r[{}{}]'.format('█' * done,
                                                   '.' * (50 - done)))
                sys.stdout.flush()
    sys.stdout.write('\n')
    assert chksum == hashlib.md5(open(
        file_path, 'rb').read()).hexdigest()[:10], 'corrupted file!'
    return file_path

def get_tokenizer(cachedir='~/kobert/'):
    """Get KoBERT Tokenizer file path after downloading
    """
    model_info = tokenizer
    return download(model_info['url'],
                    model_info['fname'],
                    model_info['chksum'],
                    cachedir=cachedir)

def get_pytorch_kobert_model(ctx='cpu', cachedir='~/kobert/'):
    # download model
    model_info = pytorch_kobert
    model_down = download(model_info['url'],
                           model_info['fname'],
                           model_info['chksum'],
                           cachedir=cachedir)
    
    zipf = ZipFile(os.path.expanduser(model_down))
    zipf.extractall(path=os.path.expanduser(cachedir))
    model_path = os.path.join(os.path.expanduser(cachedir), 'kobert_from_pretrained')
    # download vocab
    vocab_info = tokenizer
    vocab_path = download(vocab_info['url'],
                           vocab_info['fname'],
                           vocab_info['chksum'],
                           cachedir=cachedir)
    return get_kobert_model(model_path, vocab_path, ctx)


def get_kobert_model(model_path, vocab_file, ctx="cpu"):
    bertmodel = BertModel.from_pretrained(model_path)
    device = torch.device(ctx)
    bertmodel.to(device)
    bertmodel.eval()
    vocab_b_obj = nlp.vocab.BERTVocab.from_sentencepiece(vocab_file,
                                                         padding_token='[PAD]')
    return bertmodel, vocab_b_obj

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
            nn.Softmax()
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

def processing(text):
    device = torch.device("cpu")
    bertmodel, vocab = get_pytorch_kobert_model()
    tokenizer = get_tokenizer()

    model = BERTClassifier(bertmodel,batch_size=1, dr_rate=0.5).to(device)
    model.load_state_dict(torch.load("./src/model.pt", map_location=device))
    model.eval()

    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    dataset = [[text, 1]]
    data = BERTDataset(dataset, 0, 1, tok, 64, True, False)
    dataloader = torch.utils.data.DataLoader(data, batch_size = 1)

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(dataloader):
        out = model(token_ids, valid_length, segment_ids)
    
    predict_val = int(torch.max(out, 1)[1])
    print(predict_val)

if __name__ == '__main__':
    processing(sys.argv[1])
