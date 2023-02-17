import logging
import os
import os.path as osp
import subprocess
from typing import List, Optional, Union

import yaml
from mmcv.utils import Config
from modAL.models.trainval_learners import TrainValEstimator
from modAL.utils.data import modALinput
from omegaconf import DictConfig, ListConfig


class MMSegEstimator(TrainValEstimator):

    def __init__(self, config: Union[DictConfig, ListConfig]):
        self.config = config
        self.code_dir = config.train.mmseg_code_dir
        self.data_dir = config.data.root_dir
        self.gpu_count = config.train.gpu

        self.img_dir = 'leftImg8bit'
        self.ann_dir = 'gtFine'
        self.img_suffix = '_leftImg8bit.png'
        self.seg_map_suffix = '_gtFine_labelTrainIds.png'

        # generate split index file under data directory
        self.index_file = {}
        for split in ['train', 'val']:
            self.index_file[split] = f'{split}_active_learning.txt'

    def generate_active_learning_config_file(self, config_file: str, max_iters: int):
        """generate config file for active learning from origin mmseg config file

        Parameters
        ----------
        config_file : str
            mmsegmentation config file
            eg: configs/fast_scnn/fast_scnn_lr0.12_8x4_160k_cityscapes.py

        change dataset config according to self.config
        """
        cfg = Config.fromfile(config_file)
        cfg.data_root = self.data_dir

        for split in ['train', 'val']:
            cfg.data[split].data_root = self.data_dir
            cfg.data[split].split = osp.join(self.data_dir, self.index_file[split])
            cfg.data[split].img_dir = self.img_dir
            cfg.data[split].ann_dir = self.ann_dir
            cfg.data[split].img_suffix = self.img_suffix
            cfg.data[split].seg_map_suffix = self.seg_map_suffix

        if max_iters > 0:
            cfg.runner.max_iters = max_iters

        if cfg.get('work_dir', None) is None:
            # use config filename as default work_dir if cfg.work_dir is None
            cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(config_file))[0])

        new_config_file = osp.join(self.code_dir, cfg.work_dir, osp.basename(config_file))
        os.makedirs(osp.dirname(new_config_file), exist_ok=True)
        cfg.dump(new_config_file)
        return new_config_file

    def generate_split_index_file(self, split: str, X: List[str]):
        # need use the the same img_dir with the new_config_file
        img_root_dir = osp.join(self.data_dir, self.img_dir)
        with open(osp.join(self.data_dir, self.index_file[split]), 'w') as fp:
            for img_path in X:
                # use the same img_suffix with the new_config_file
                img_name = osp.relpath(img_path, img_root_dir).replace(self.img_suffix, '')
                fp.write(img_name + '\n')

    def fit(self, X_train: List[str], y_train: List[str], X_val: List[str], y_val: List[str], **fit_kwargs) -> None:
        assert len(X_train) == len(y_train)
        assert len(X_val) == len(y_val)

        self.generate_split_index_file('train', X_train)
        self.generate_split_index_file('val', X_val)

        config_file = self.generate_active_learning_config_file(osp.join(self.code_dir, self.config.model.config),
                                                                max_iters=30 * len(X_train))

        train_cmd = f'bash ./tools/dist_train.sh {config_file} {self.gpu_count}'

        logging.info(f'run cmd: {train_cmd} under {self.code_dir}')
        subprocess.run(train_cmd.split(), check=True, cwd=self.code_dir)

    def score(self, X: Optional[List[str]] = None, y: Optional[List[str]] = None, **score_kwargs):
        return 0

    def predict(self, X):
        pass

    def predict_proba(self, X):
        return [self.model.predict(x) for x in X]
