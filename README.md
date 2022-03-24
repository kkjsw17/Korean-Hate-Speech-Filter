# Hate Speech Filter
This project was conducted in 2021-1 semester as part of Konkuk univ. Open Source SW Project 1 project.\
It functions as a discriminator whether sentence that client inputs is a hate-speech, using KoBERT.
## Getting Started Hate-Speech-Filter
### 1. install node modules
```
npm i
```
### 2. install python modules
```
pip install -r requirements.txt
```
### 3. get model checkpoint file (for load)
```
python src/server/get_model.py
```
### 4. run flask server
```
python src/server/app.py
```
### 5. run front-end
```
npm start
```

## Architecture
### 1. BERT
![KakaoTalk_20210908_220746731](https://user-images.githubusercontent.com/39490214/132515385-4b2d0325-dbfd-45c3-974e-d6ef8e72b554.png)
### 2. BERT + CNN
![KakaoTalk_20210908_220754320](https://user-images.githubusercontent.com/39490214/132515628-62e0d2b3-5267-4d62-b6f0-2fc29400a984.png)
### 3. BERT + Ensembled CNN
![KakaoTalk_20210908_220804347](https://user-images.githubusercontent.com/39490214/132515733-2366470f-94d7-4ce2-b838-adff55e6b38c.png)
### 4. BERT + Ensembled CNN + BiLSTM
![KakaoTalk_20210908_220822686](https://user-images.githubusercontent.com/39490214/132515781-6d5f029c-1e0a-42e3-b14e-61842841eac3.png)

## Test Accuracy
![image](https://user-images.githubusercontent.com/77087144/132781502-f1edb88e-ca56-4207-82f2-f8914468ea87.png)

# Reference
### KoBART
https://github.com/SKT-AI/KoBART
### Korean Hate Speech Dataset
https://github.com/kocohub/korean-hate-speech/blob/master/labeled/train.tsv \
https://github.com/ZIZUN/korean-malicious-comments-dataset

