"""
Author: songjun
Date: 2018/4/17
Description:
Usage:
"""
from ml_idiot.evaluator.CategoricalEvaluator import CategoricalEvaluator


class BaseTrainer(object):
    def __init__(self):
        self.metric = ['accuracy']
        self.evaluator = CategoricalEvaluator(self.metric)

    def evaluate_metrics(self, gth_labels, pred_labels):
        self.evaluator.evaluate(gth_labels, pred_labels)
