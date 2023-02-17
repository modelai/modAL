import os.path as osp
from typing import List, Union

import yaml
from modAL.models.base import BaseLearner
from modAL.utils.data import modALinput
from omegaconf import DictConfig, ListConfig, OmegaConf
from sklearn.base import BaseEstimator
from ultralytics import YOLO


class Yolov5Estimator(BaseEstimator):

    def __init__(self, config: Union[DictConfig, ListConfig]):
        # Load a model
        self.config = config

        # self.model = YOLO("yolov8n.yaml")  # build a new model from scratch
        # self.model = YOLO(conf.model.weight)  # load a pretrained model (recommended for training)

        data_yaml = osp.join(self.config.data.path, 'init_active_learning.yaml')
        OmegaConf.save(config=config.data, f=data_yaml)
        # self.model = YOLO(conf.model.yaml, type='v6.1', data=data_yaml)
        self.model = YOLO(config.model.weight)

    def fit(self, X: List[str], y: List[str], bootstrap: bool = False, **fit_kwargs) -> None:
        train_txt = osp.join(self.config.data.path, 'train_active_learning.txt')
        self.config.data.train = train_txt
        with open(train_txt, 'w') as fp:
            for img_path in X:
                fp.write(f'{img_path}\n')

        data_yaml = osp.join(self.config.data.path, 'fit_active_learning.yaml')
        OmegaConf.save(config=self.config.data, f=data_yaml)

        # self.conf.data.val = 'runs/val.txt'
        self.model.train(data=data_yaml, epochs=3, conf=0.1, project=osp.abspath('./runs'))  # train the model

    def score(self, X, y):
        metrics = self.model.val()  # evaluate model performance on the validation set
        return metrics

    def predict(self, X):
        return [self.model.predict(x) for x in X]

    def predict_proba(self, X):
        return [self.model.predict(x) for x in X]

    def predict_featuremap(self, X):
        pass
