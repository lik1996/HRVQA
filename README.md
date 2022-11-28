# HRVQA
Implementation for the paper "HRVQA: A Visual Question Answering Benchmark for High-Resolution Aerial Images". 
---

We benchmark an aerial image visual question answering task with our proposed dataset HRVQA, and more details about this dataset could be found in our official website [HRVQA](https://uavid.nl/). The evaluation server and the benchmark table are held on [Codalab](https://uavid.nl/) platform. Welcome to submit your results!



### Preparations
---
&emsp;1. [Cuda](https://developer.nvidia.com/zh-cn/cuda-toolkit) and [Cudnn](https://developer.nvidia.com/cudnn).


&emsp;2. Conda enviroment:

```
    $ conda create -n vqa python==3.8.10
    $ source activate vqa
```


&emsp;3. Install pytorch(1.8.1) and torchvision(0.9.1):

```
    $ pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

&emsp;4. Install [SpaCy](https://spacy.io/) and initialize the [GloVe](https://github.com/explosion/spacy-models/releases/download/en_vectors_web_lg-2.1.0/en_vectors_web_lg-2.1.0.tar.gz) as follows:

```
    $ wget https://github.com/explosion/spacy-models/releases/download/en_vectors_web_lg-2.1.0/en_vectors_web_lg-2.1.0.tar.gz -O en_vectors_web_lg-2.1.0.tar.gz
    $ pip install en_vectors_web_lg-2.1.0.tar.gz
```


### Dataset Download
---
#### a. Visual Input Download
The aerial images can be downloaded in our official website [HRVQA](https://uavid.nl/). To make it convenient, we also provide the grid visual features for training and inference stages. You can find the files in [hrvqa-visual-features](https://uavid.nl/): train, val, test.

#### b. Lingual Input Download
The question-answer pairs you can download in this [HRVQA](https://uavid.nl/). Downloaded files contains in a folder named jsons: train_question, train_answer, val_question, val_answer, test_question.

More metadata information could be found in the [HRVQA](https://uavid.nl/).


### Training
---

The following script will start training with the default hyperparameters:

```
$ python run.py --RUN=train --METHOD=oursv1 --VERSION='hrvqa_grid_glove'
```

All checkpoint files will be saved to:
```
ckpts/ckpt_<VERSION>/epoch<EPOCH_NUMBER>.pkl
```

and the training log file will be placed at:
```
results/log/log_run_<VERSION>.txt
```


### Val and Test
---

The following script will start valing with the default hyperparameters:
```
$ python run.py --RUN=val --CKPT_PATH=generated_CKPT_PATH --METHOD=oursv1 --VERSION='hrvqa_grid_glove'
```


The following script will start testing with the default hyperparameters:
```
$ python run.py --RUN=test --CKPT_PATH=generated_CKPT_PATH --METHOD=oursv1 --VERSION='hrvqa_grid_glove'
```
You can find the test result in:
```
/results/result_test/result_run_<'VERSION+EPOCH'>.json
```

### Models 
---
Here we provide our pre-trained models in this [link](https://uavid.nl/).


### Result 
---
Following this steps you should be able to reproduce the results in the paper. The performance of the propopsed method on test split is reported as follows:

|  Number   | Yes/No | Areas |  Size |  Locat. | Color | Shape | Sports | Trans. | Scene | OA | AA |
|  ----  | ----  | ----  |----  |----  |----  |----  |----  |----  |----  |----  |----  |
| 66.50 | 93.32 | 97.11 | 93.72 | 74.82 | 45.36 | 96.67 | 77.03 | 88.87 | 77.30 | 81.71 | 81.07 | 

---

### Citation
---
if HRVQA is helpful for your research or you wish to refer the baseline results published here, we'd really appreciate it if you could cite this paper.
```
@InProceedings{
    author    = {Kun Li, George Vosselman and Michael Ying Yang},
    title     = {HRVQA: A Visual Question Answering Benchmark for High-Resolution Aerial Images},
    year      = {2022},
}
```

**Here, we thanks so much for these great works:  [mcan-vqa](https://github.com/MILVLG/mcan-vqa) and [TRAR-VQA](https://github.com/rentainhe/TRAR-VQA)** 
