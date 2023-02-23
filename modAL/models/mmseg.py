import glob
import json
import logging
import os
import os.path as osp
import subprocess
import sys
from typing import List, Optional, Union

import mmcv
import numpy as np
import torch
from mmcv.parallel import collate, scatter
from mmcv.utils import Config
from modAL.models.trainval_learners import TrainValEstimator
from omegaconf import DictConfig, ListConfig, OmegaConf


class LoadImage:
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """

        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


class MMSegEstimator(TrainValEstimator):
    """openmmlab/mmsegmentation estimator, test on v0.30.0

    input config:
    code_dir: the mmsegmentation code directory
    data_dir: the cityscapes dataset directory
    max_epoched/max_iters: the training iteration number
    project_name: the save directory tag

    internal variable:
    self.work_dir: training result save directory
        - set after training function self.fit()
        - used in evaluation function self.score()
    self.last_weight_file: last saved weight file
        - set after training function self.fit()
        - used in the next training iteration as init weight file
    self.index_file: the train/val dataset index file
        - generate before training command `train_cmd`
        - set in self.generate_active_learning_config_file()

    output:
    self.predict(): the prediction result, after argmax
    self.predict_proba(): the prediction feature map, before argmax
    self.score(): the evaluation result
    """

    def __init__(self, config: Union[DictConfig, ListConfig]):
        self.config = config
        self.code_dir = config.train.mmseg_code_dir

        self.data_dir = config.data.root_dir
        self.gpu_count = config.train.gpu

        self.img_dir = 'leftImg8bit'
        self.ann_dir = 'gtFine'
        self.img_suffix = '_leftImg8bit.png'
        self.seg_map_suffix = '_gtFine_labelTrainIds.png'

        self.work_dir = ''
        self.last_weight_file = ''
        self.model = None
        # generate split index file under data directory
        self.index_file = {}
        for split in ['train', 'val']:
            self.index_file[split] = f'{split}_active_learning.txt'

    def generate_active_learning_config_file(self, config_file: str, max_iters: int, project_name: str = ''):
        """generate config file for active learning from origin mmseg config file

        Parameters
        ----------
        config_file : str
            mmsegmentation config file
            eg: configs/fast_scnn/fast_scnn_lr0.12_8x4_160k_cityscapes.py
        project_Name: str
            project name
            eg: random_0, rsal_3

        change dataset config according to self.config
        """
        cfg = Config.fromfile(config_file)
        cfg.data_root = self.data_dir

        for split in ['train', 'val']:
            cfg.data[split].data_root = self.data_dir
            # use the target mining dataset
            cfg.data[split].split = osp.join(self.data_dir, self.index_file[split])
            cfg.data[split].img_dir = self.img_dir
            cfg.data[split].ann_dir = self.ann_dir
            cfg.data[split].img_suffix = self.img_suffix
            cfg.data[split].seg_map_suffix = self.seg_map_suffix

        if max_iters > 0:
            cfg.runner.max_iters = max_iters
            cfg.checkpoint_config.interval = max(1, max_iters // 3)
            cfg.evaluation.interval = max(1, max_iters // 3)

        if cfg.get('work_dir', None) is None:
            # use config filename as default work_dir if cfg.work_dir is None
            cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(config_file))[0], project_name)

        # add tensorboard log, but numpy >=1.24 cause np.object error
        # cfg.log_config.hooks.append(dict(type='TensorboardLoggerHook', by_epoch=False))

        # load pretrained weight
        if self.last_weight_file:
            cfg.load_from = self.last_weight_file

        new_config_file = osp.join(self.code_dir, cfg.work_dir, osp.basename(config_file))
        os.makedirs(osp.dirname(new_config_file), exist_ok=True)
        cfg.dump(new_config_file)

        self.work_dir = osp.join(self.code_dir, cfg.work_dir)
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

        if 'max_iters' in fit_kwargs:
            max_iters = fit_kwargs['max_iters']
        elif 'max_epoches' in fit_kwargs:
            max_epoches = fit_kwargs['max_epoches']
            max_iters = max_epoches * len(X_train)
        else:
            assert False
            max_iters = 0

        if 'project_name' in fit_kwargs:
            project_name = fit_kwargs['project_name']
        else:
            assert False
            project_name = ''

        self.generate_split_index_file('train', X_train)
        self.generate_split_index_file('val', X_val)

        config_file = self.generate_active_learning_config_file(osp.join(self.code_dir, self.config.model.config),
                                                                max_iters=max_iters,
                                                                project_name=project_name)

        train_cmd = f'bash ./tools/dist_train.sh {config_file} {self.gpu_count}'

        logging.info(f'run cmd: {train_cmd} under {self.code_dir}')

        weight_files = glob.glob(osp.join(self.work_dir, 'iter_*.pth'))
        if len(weight_files) == 0:
            # skip trained class
            subprocess.run(train_cmd.split(), check=True, cwd=self.code_dir)

        weight_files = glob.glob(osp.join(self.work_dir, 'iter_*.pth'))
        self.last_weight_file = max(weight_files, key=osp.getctime)
        logging.info(f'last weight file saved on {self.last_weight_file}')

    def score(self, X: Optional[List[str]] = None, y: Optional[List[str]] = None, **score_kwargs):
        """load training log file to obtain evaluation result
        """
        log_files = glob.glob(osp.join(self.work_dir, '*.log.json'))
        assert len(log_files) > 0, f'no log file found on {self.work_dir}'
        newest_log_file = max(log_files, key=osp.getctime)

        with open(newest_log_file, 'r') as fp:
            lines = fp.readlines()

        for line in reversed(lines):
            metrics = json.loads(line.strip())

            if metrics['mode'] == 'val':
                return metrics

        assert False, f'no validation results found in {newest_log_file}'

    def predict(self, X: List[str], device: str = 'cuda:0'):
        if self.model is None:
            sys.path.append(self.code_dir)
            from mmseg.apis import inference_segmentor, init_segmentor
            config_file = glob.glob(osp.join(self.work_dir, '*.py'))[0]
            self.model = init_segmentor(config_file, self.last_weight_file, device=device)

        return [inference_segmentor(self.model, x) for x in X]

    def predict_proba(self, X: Union[str, List[str]], device: str = 'cuda:0'):
        if self.model is None:
            sys.path.append(self.code_dir)
            from mmseg.apis import init_segmentor
            config_file = glob.glob(osp.join(self.work_dir, '*.py'))[0]
            self.model = init_segmentor(config_file, self.last_weight_file, device=device)

        return self.inference(X)

    def inference(self, imgs: Union[str, List[str]]) -> List[np.ndarray]:
        assert self.model is not None, 'please init self.model first'
        from mmseg.datasets.pipelines import Compose

        cfg = self.model.cfg
        device = next(self.model.parameters()).device  # model device
        # build the data pipeline
        test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
        test_pipeline = Compose(test_pipeline)
        # prepare data
        data = []
        imgs = imgs if isinstance(imgs, list) else [imgs]
        for img in imgs:
            img_data = dict(img=img)
            img_data = test_pipeline(img_data)
            data.append(img_data)
        data = collate(data, samples_per_gpu=len(imgs))
        if next(self.model.parameters()).is_cuda:
            # scatter to specified GPU
            data = scatter(data, [device])[0]
        else:
            data['img_metas'] = [i.data[0] for i in data['img_metas']]

        results = []
        for idx in range(len(data['img'])):
            with torch.no_grad():
                result = self.model.inference(img=data['img'][idx], img_meta=data['img_metas'][idx], rescale=True)

                results.append(result.data.cpu().numpy())
        return results

    def dump_omega_config(self):
        """dump omega config for mining
        set config_file and checkpoint for mmsegmentation mining
        """
        self.config.mining.config_file = glob.glob(osp.join(self.work_dir, '*.py'))[0]
        self.config.mining.checkpoint = self.last_weight_file
        OmegaConf.save(config=self.config, f=self.config.mining.omega_config_file)

        return self.config.mining.omega_config_file
