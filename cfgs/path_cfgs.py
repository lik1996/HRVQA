# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

import os

class PATH:
    def __init__(self):

        # hrvqa dataset root path
        self.DATASET_PATH = '/home/lik/data/VHR-QA/'

        # features root path
        self.FEATURE_PATH = '/home/lik/data/VHR-QA/features 1024/'  # changed for 1024

        self.PRETRAINED_PATH = '/home/lik/code/pretrained/bert_base_uncased/'

        self.init_path()


    def init_path(self):

        self.IMG_FEAT_PATH = {
            'train': self.FEATURE_PATH + 'train/',
            'val': self.FEATURE_PATH + 'val/',
            'test': self.FEATURE_PATH + 'test/',
        }

        self.QUESTION_PATH = {
            'train': self.DATASET_PATH + 'jsons/mutan/' + 'train_question.json',
            'val': self.DATASET_PATH + 'jsons/mutan/' + 'val_question.json',
            'test': self.DATASET_PATH + 'jsons/mutan/' + 'test_question.json',
        }

        self.ANSWER_PATH = {
            'train': self.DATASET_PATH + 'jsons/mutan/' + 'train_answer.json',
            'val': self.DATASET_PATH + 'jsons/mutan/' + 'val_answer.json',
        }

        self.RESULT_PATH = './results/result_test/'
        self.PRED_PATH = './results/pred/'
        self.CACHE_PATH = './results/cache/'
        self.LOG_PATH = './results/log/'
        self.CKPTS_PATH = './ckpts/'

        if 'result_test' not in os.listdir('./results'):
            os.mkdir('./results/result_test')

        if 'pred' not in os.listdir('./results'):
            os.mkdir('./results/pred')

        if 'cache' not in os.listdir('./results'):
            os.mkdir('./results/cache')

        if 'log' not in os.listdir('./results'):
            os.mkdir('./results/log')

        if 'ckpts' not in os.listdir('./'):
            os.mkdir('./ckpts')


    def check_path(self):
        print('Checking dataset ...')

        for mode in self.IMG_FEAT_PATH:
            if not os.path.exists(self.IMG_FEAT_PATH[mode]):
                print(self.IMG_FEAT_PATH[mode] + 'NOT EXIST')
                exit(-1)

        for mode in self.QUESTION_PATH:
            if not os.path.exists(self.QUESTION_PATH[mode]):
                print(self.QUESTION_PATH[mode] + 'NOT EXIST')
                exit(-1)

        for mode in self.ANSWER_PATH:
            if not os.path.exists(self.ANSWER_PATH[mode]):
                print(self.ANSWER_PATH[mode] + 'NOT EXIST')
                exit(-1)

        print('Finished')
        print('')

