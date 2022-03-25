# Korean Hate Speech Filter
This project was conducted in 2021-1 semester as part of Konkuk univ. Open Source SW Project 1 project.\
It functions as a discriminator whether sentence that client inputs is a hate-speech, using KoBERT.
![banner](https://user-images.githubusercontent.com/39583312/160071672-af0e628c-d27c-41fd-b52c-7c80c0181dbe.png)

* [Hate Speech Filter](#hate-speech-filter)
  * [How to install](#how-to-install)
    * [Clone project and install modules](#clone-project-and-install-modules)
    * [Get model checkpoint file](#get-model-checkpoint-file-for-load)
  * [Getting Started](#getting-started)
    * [Run flask server](#run-flask-server)
    * [Run front-end](#run-front-end)
  * [Demo](#demo)
  * [Model Architecture](#model-architecture)
  * [Test Accuracy](#test-accuracy)
  * [Reference](#reference)
    * [KoBERT](#kobert)
    * [Data](#data)

---

## How to install

### Clone project and install modules
```
git clone https://github.com/kkjsw17/Korean-Hate-Speech-Filter.git
cd Korean-Hate-Speech-Filter
npm i
pip install -r requirements.txt
```

### Get model checkpoint file (for load)
```
python src/server/get_model.py
```

## Getting started

### Run flask server
```
python src/server/app.py
```

### Run front-end
```
npm start
```

## Demo
> Please click to see image bigger
<table>
  <tr>
   <td align=center>Pass</td>
   <td align=center>Detection</td>
  </tr>
  <tr>
    <td><img src="demo/1.png" alt="1" ></td>
    <td><img src="demo/3.png" alt="2" ></td>
  </tr>
  <tr>
    <td><img src="demo/2.png" alt="3" ></td>
    <td><img src="demo/4.png" alt="4" ></td>
  </tr>
</table>


## Model Architecture

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

## Reference

### KoBERT
https://github.com/SKTBrain/KoBERT

### Data
1. [Korean HateSpeech Dataset, ](https://github.com/kocohub/korean-hate-speech/blob/master/labeled/train.tsv)
2. [korean-malicious-comments-dataset](https://github.com/ZIZUN/korean-malicious-comments-dataset)
