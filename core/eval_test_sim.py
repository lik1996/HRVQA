# --------------------------------------------------------
# hrvqa
# Licensed under The MIT License [see LICENSE for details]
# Written by lik
# --------------------------------------------------------
# evaluate result.json generated from test model

import json
import os
import numpy as np

def question_type_json2dic(json2):
    with open(json2) as json_data:
        questionsJSON = json.load(json_data)

    que_dic = {}

    for ques in questionsJSON['questions']:
        question_id = ques['question_id']
        question_type = ques['question_type']

        if not question_id in que_dic:
            if question_type == 'numbers':
                que_dic[question_id] = 0
            elif question_type == 'yes no':
                que_dic[question_id] = 1
            elif question_type == 'areas':
                que_dic[question_id] = 2
            elif question_type == 'size':
                que_dic[question_id] = 3
            elif question_type == 'location':
                que_dic[question_id] = 4
            elif question_type == 'color':
                que_dic[question_id] = 5
            elif question_type == 'shape':
                que_dic[question_id] = 6
            elif question_type == 'sports':
                que_dic[question_id] = 7
            elif question_type == 'transportation':
                que_dic[question_id] = 8
            elif question_type == 'scene':
                que_dic[question_id] = 9

    return que_dic

def answer_str_change(str1):
    ans_proc = ''
    if str1[-2:] == 'm2':
        value = float(str1[:-2])
        if value > 0.0 and value <= 10.0:
            ans_proc = "between 0m2 and 10m2"
        if value > 10.0 and value <= 100.0:
            ans_proc = "between 10m2 and 100m2"
        if value > 100. and value <= 1000.:
            ans_proc = "between 100m2 and 1000m2"
        if value > 1000.0:
            ans_proc = "more than 1000m2"
        else:
            ans_proc = "0m2"
    elif str1 == 'fitnessorsoccer':
        ans_proc = 'fitness or soccer'
    elif str1 == 'fitnessorbasketball':
        ans_proc = 'fitness or basketball'
    elif str1 == 'tennisorsoccer':
        ans_proc = 'tennis or soccer'
    elif str1 == 'basketballorsoccer':
        ans_proc = 'basketball or soccer'
    else:
        ans_proc = str1
    return ans_proc

def json4comparison(json1, json2):
    with open(json1) as json_data:
        answersJSON = json.load(json_data)

    with open(json2) as json_data2:
        predsJSON = json.load(json_data2)

    acc_type = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    num_type = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    right_pred = 0
    overall_acc = 0.0
    average_acc = 0.0

    ques_type_dic = question_type_json2dic(os.path.join('/home/lik/data/VHR-QA/', 'jsons', 'mutan', 'test_question.json'))

    for t, ans in enumerate(answersJSON['annotations']):
        question_id = ans['question_id']
        answer_str = answer_str_change(ans['multiple_choice_answer'])
        # for k in enumerate(predsJSON):

        pred_answer_str = predsJSON[t]['answer']

        for i in range(10):
            if i == ques_type_dic[question_id]:
                num_type[i] += 1
                if answer_str == pred_answer_str:
                    right_pred += 1
                    acc_type[i] += 1
                break

    target = open('/home/lik/code/vhrqa/results/txts/test_scores_epoch13_oursv1_v3_512.txt', "w")
    question_type_list = ['numbers', 'yes', 'areas', 'size', 'location', 'color', 'shape', 'sports', 'transportation', 'scene']
    overall_acc = right_pred / len(predsJSON)

    for t in range(10):
        target.write("%s: %.4f\n" % (question_type_list[t], (acc_type[t] / num_type[t])))
        average_acc += acc_type[t] / num_type[t]

    target.write("oa: %.4f aa: %.4f\n" % (overall_acc, (average_acc / 10)))
    target.close()


if __name__ == '__main__':
    json1 = os.path.join('/home/lik/data/VHR-QA/', 'jsons', 'mutan', 'test_answer.json')
    json2 = os.path.join('/home/lik/code/vhrqa/results/result_test', 'result_run_epoch13.pkl_oursv1_v3_512.json')

    json4comparison(json1, json2)

    print("all work is done!\n")