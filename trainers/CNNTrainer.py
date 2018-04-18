"""
Author: songjun
Date: 2018/4/17
Description:
Usage:
"""
from trainers.BaseTrainer import BaseTrainer


class CNNTrainer(BaseTrainer):
    def __init__(self):
        super(CNNTrainer, self).__init__()

    def train_model(self, model, data_provider, config):
        pass

    def test_model(self, model, data_provider, config):
        pass