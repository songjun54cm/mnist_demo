__author__ = 'JunSong<songjun54cm@gmail.com>'
import argparse
import importlib
import os
import logging
import json
from ml_idiot.solver.BasicSolver import BasicSolver as Solver


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"


def create_data_provider(config):
    dpname = '%sDataProvider' % config['data_set_name']
    DPClass = getattr(importlib.import_module('data_providers'), dpname)
    data_provider = DPClass()
    data_provider.create()
    return data_provider


def create_model(config):
    ModelClass = getattr(importlib.import_module('models'), '%sModel'%config['model_name'])
    model = ModelClass()
    model.create_model()
    model.create(config['model_config'])
    return model


def create_trainer(config):
    TrainerClass = getattr(importlib.import_module('trainers'), config['trainer_name'])
    trainer = TrainerClass()
    return trainer


def complete_config(config):
    if 'trainer_name' not in config:
        config['trainer_name'] = '%sTrainer' % config['model_name']
    if 'output_dir' not in config:
        suffix = 1
        output_dir = os.path.join(DataHome, 'output', config['model_name'], config['data_set_name'], str(suffix))
        if os.path.exists(output_dir):
            suffix += 1
            output_dir = os.path.join(DataHome, 'output', config['model_name'], config['data_set_name'], str(suffix))
        config['output_dir'] = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    if config['learning_rate'] is not None:
        config['model_config']['learning_rate'] = config['learning_rate']
    else:
        config['learning_rate'] = config['model_config']['learning_rate']


def set_up_logger(config):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(filename)s : %(levelname)s  %(message)s',
        datefmt='%Y-%m-%d %A %H:%M:%S',
        filename=os.path.join(config['output_dir'], 'log.log'),
        filemode='w'
    )
    console = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s  %(filename)s : %(levelname)s  %(message)s')
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def main(config):
    complete_config(config)
    logging.info(json.dumps(config, indent=2))
    data_provider = create_data_provider(config)
    model = create_model(config)
    trainer = create_trainer(config)
    trainer.train_model(model, data_provider, config)
    trainer.test_model(model, data_provider)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', dest='data_set_name', type=str, default='MNIST')
    parser.add_argument('-m', '--model', dest='model_name', type=str, default='CNN')
    parser.add_argument('-f', '--file', dest='config_file', type=str, default=None)
    args = parser.parse_args()
    config = vars(args)
    if config['config_file'] is None:
        config['config_file'] = '%s_%s_Config' % (config['data_set_name'], config['model_mane'])
    config.update(getattr(importlib.import_module('configures.%s' % config['config_file']), 'config'))
    main(config)
