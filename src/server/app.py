from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import torch
import gluonnlp as nlp
from kobert.pytorch_kobert import get_pytorch_kobert_model, get_tokenizer
from BERT import BERTClassifier, BERTDataset

app = Flask(__name__)
CORS(app, resources={r'/*': {'origins': '*'}})

device = torch.device("cpu")
bertmodel, vocab = get_pytorch_kobert_model()
tokenizer = get_tokenizer()

model = BERTClassifier(bertmodel, batch_size=1).to(device)
saved_checkpoint = torch.load(".model/model.pt", map_location=device)
model.load_state_dict(saved_checkpoint, strict=False)

@app.route('/', methods=['POST', 'GET'])
def send_result():
    if request.method == 'POST':
        comment = json.loads(request.get_data())['comment']
        prediction = predict(comment)
        return jsonify(prediction)
    else:
        return jsonify('Invalid Methods')

def predict(text):
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    dataset = [[text, 1]]
    data = BERTDataset(dataset, 0, 1, tok, 64, True, False)
    dataloader = torch.utils.data.DataLoader(data, batch_size = 1)

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(dataloader):
        out = model(token_ids, valid_length, segment_ids)
    
    predict_val = int(torch.max(out, 1)[1])
    return predict_val

if __name__ == '__main__':
    app.run(debug=True)