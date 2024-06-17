# HRVQA
Implementation for the paper "HRVQA: A Visual Question Answering Benchmark for High-Resolution Aerial Images". 
---

We benchmark an aerial image visual question answering task with our proposed dataset HRVQA, and more details about this dataset can be found on our official website [HRVQA](https://hrvqa.nl/). The evaluation server and the benchmark table will be held on [Codalab](https://codalab.lisn.upsaclay.fr/) platform. Welcome to submit your results!



### Preparations
---
&emsp;1. [Cuda](https://developer.nvidia.com/zh-cn/cuda-toolkit) and [Cudnn](https://developer.nvidia.com/cudnn).


&emsp;2. Conda environment:

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
The aerial images can be downloaded from our official website [HRVQA](https://hrvqa.nl/). To make it convenient, we also provide the grid visual features for the training and inference stages. You can find the files in [hrvqa-visual-features](https://hrvqa.nl/): train, val, test.

#### b. Lingual Input Download
The question-answer pairs can be downloaded in this [HRVQA](https://hrvqa.nl/). Downloaded files are contained in a folder named jsons: train_question, train_answer, val_question, val_answer, test_question.

More metadata information can be found in the [HRVQA](https://hrvqa.nl/).


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

The following script will start validation with the default hyperparameters:
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
The pre-trained models are coming soon.
<!-- 
[GFTransformer-ED](https://drive.google.com/uc?export=download&id=1bkyJ6Ilz-wq92TDSV0yeeu8-n75r0tMS) means encoder-decoder architecture.
[GFTransformer-S](https://drive.google.com/uc?export=download&id=1t6k3zBUNyq_bd6ujGw-PZvldSVvDG71W) means stacked architecture.
-->

### Result 
---
Following these steps, you should be able to reproduce the results in the paper. The performance of the propopsed method on test split is reported as follows:

|  Number   | Y/N | Areas |  Size |  Locat. | Color | Shape | Sports | Trans. | Scene | OA | AA |
|  ----  | ----  | ----  |----  |----  |----  |----  |----  |----  |----  |----  |----  |
| 66.50 | 93.32 | 97.11 | 93.72 | 74.82 | 45.36 | 96.67 | 77.03 | 88.87 | 77.30 | 81.71 | 81.07 | 

---

### Citation
---
If HRVQA is helpful for your research or you wish to refer to the baseline results published here, we'd appreciate it if you could cite this paper.
```
@article{LI202465,
title = {HRVQA: A Visual Question Answering benchmark for high-resolution aerial images},
journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
volume = {214},
pages = {65-81},
year = {2024},
issn = {0924-2716},
doi = {https://doi.org/10.1016/j.isprsjprs.2024.06.002},
author = {Kun Li and George Vosselman and Michael Ying Yang}
}
```

Here, we thank so much for these great works:  [mcan-vqa](https://github.com/MILVLG/mcan-vqa) and [TRAR-VQA](https://github.com/rentainhe/TRAR-VQA)
