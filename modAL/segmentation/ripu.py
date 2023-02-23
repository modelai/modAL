"""
Towards Fewer Annotations: Active Learning via Region Impurity and
    Prediction Uncertainty for Domain Adaptive Semantic Segmentation (CVPR 2022 Oral)

view code: https://github.com/BIT-DA/RIPU
"""

import argparse
import os
import os.path as osp
import sys
from typing import Dict, List, Union

import mmcv
import torch
import torch.distributed as dist
import torch.nn.functional as F
from mmcv.engine import collect_results_cpu
from mmcv.runner import init_dist, wrap_fp16_model
from modAL.segmentation.utils import get_dataloader
from omegaconf import DictConfig, ListConfig, OmegaConf
from tqdm import tqdm

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


class RIPUMining(torch.nn.Module):

    def __init__(self, cfg: Union[DictConfig, ListConfig], class_number: int):
        super().__init__()
        self.cfg = cfg
        self.region_radius = int(cfg.mining.ripu_region_radius)
        self.class_number = class_number
        self.image_topk = int(cfg.mining.topk_superpixel_score)
        # ratio = float(ymir_cfg.param.ratio)

        kernel_size = 2 * self.region_radius + 1
        self.region_pool = torch.nn.AvgPool2d(kernel_size, stride=1, padding=self.region_radius)

        self.depthwise_conv = torch.nn.Conv2d(in_channels=self.class_number,
                                              out_channels=self.class_number,
                                              kernel_size=kernel_size,
                                              stride=1,
                                              padding=self.region_radius,
                                              bias=False,
                                              padding_mode='zeros',
                                              groups=self.class_number)

        weight = torch.ones((self.class_number, 1, kernel_size, kernel_size), dtype=torch.float32)
        weight = torch.nn.Parameter(weight)
        self.depthwise_conv.weight = weight
        self.depthwise_conv.requires_grad_(False)

    def get_region_uncertainty(self, logit: torch.Tensor) -> torch.Tensor:
        C = torch.tensor(logit.shape[1])
        entropy = -logit * torch.log(logit + 1e-6)  # BCHW
        uncertainty = torch.sum(entropy, dim=1, keepdim=True) / torch.log(C)  # B1HW

        region_uncertainty = self.region_pool(uncertainty)  # B1HW

        return region_uncertainty

    def get_region_impurity(self, logit: torch.Tensor) -> torch.Tensor:
        C = torch.tensor(logit.shape[1])
        predict = torch.argmax(logit, dim=1)  # BHW
        one_hot = F.one_hot(predict, num_classes=self.class_number).permute(
            (0, 3, 1, 2)).to(torch.float)  # BHW --> BHWC --> BCHW
        summary = self.depthwise_conv(one_hot)  # BCHW
        count = torch.sum(summary, dim=1, keepdim=True)  # B1CH
        dist = summary / count  # BCHW
        region_impurity = torch.sum(-dist * torch.log(dist + 1e-6), dim=1, keepdim=True) / torch.log(C)  # B1HW

        return region_impurity

    def get_region_score(self, logit: torch.Tensor) -> torch.Tensor:
        """
        logit: [B,C,H,W] prediction result with softmax/sigmoid
        """
        score = self.get_region_uncertainty(logit) * self.get_region_impurity(logit)  # B1HW

        return score

    def get_image_score(self, logit: torch.Tensor) -> torch.Tensor:
        B, C, H, W = logit.shape
        score = self.get_region_score(logit).view(size=(B, 1 * H * W))  # B1HW
        topk = torch.topk(score, k=self.image_topk, dim=1, largest=True)  # BK
        image_score = torch.sum(topk.values, dim=1)  # B
        return image_score


def iter_fun(cfg, model, miner, idx, batch) -> List[Dict]:
    """
    batch: Dict(img=[tensor,], img_metas=[List[DataContainer],])
    """
    batch_img = batch['img'][0]
    batch_meta = batch['img_metas'][0].data[0]
    device = torch.device('cuda:0' if RANK == -1 else f'cuda:{RANK}')
    batch_result = model.inference(batch_img.to(device), batch_meta, rescale=True)

    scores = miner.get_image_score(batch_result)
    image_results: List[Dict] = []
    for idx, meta in enumerate(batch_meta):
        score = float(scores[idx])
        image_results.append(dict(image_filepath=meta['filename'], score=score))

    return image_results


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--omega_config', required=True, help='omega config file')
    parser.add_argument('--index_file', required=True, help='the input mining index file')
    parser.add_argument('--out_file', required=True, help='the output result')
    return parser.parse_args()


def main() -> int:
    if LOCAL_RANK != -1:
        init_dist(launcher='pytorch', backend="nccl" if dist.is_nccl_available() else "gloo")

    args = get_args()

    cfg = OmegaConf.load(args.omega_config)
    code_dir = cfg.train.mmseg_code_dir
    sys.path.append(code_dir)
    from mmseg.apis import init_segmentor

    # mmsegmentation config file
    config_file = cfg.mining.config_file
    checkpoint_file = cfg.mining.checkpoint

    index_file_path = args.index_file
    with open(index_file_path, 'r') as f:
        origin_image_list = [line.strip() for line in f.readlines()]
        num_image_all_rank = len(origin_image_list)

    samples_per_gpu = int(cfg.mining.samples_per_gpu)
    max_barrier_times = num_image_all_rank // WORLD_SIZE // samples_per_gpu
    mmcv_cfg = mmcv.Config.fromfile(config_file)
    mmcv_cfg.model.train_cfg = None
    model = init_segmentor(config=mmcv_cfg,
                           checkpoint=checkpoint_file,
                           device='cuda:0' if RANK == -1 else f'cuda:{RANK}')
    class_num = len(cfg.data.class_names)
    miner = RIPUMining(cfg, class_num)
    miner.to('cuda:0' if RANK == -1 else f'cuda:{RANK}')

    if cfg.mining.fp16:
        wrap_fp16_model(model)

    dataloader = get_dataloader(mmcv_cfg, cfg, index_file_path)
    N = len(dataloader)
    if N == 0:
        raise Exception('find empty dataloader')

    if RANK in [0, -1]:
        tbar = tqdm(dataloader, desc='obtain image prediction')
    else:
        tbar = dataloader

    rank_image_result = []
    for idx, batch in enumerate(tbar):
        if idx < max_barrier_times and WORLD_SIZE > 1:
            dist.barrier()
        batch_image_result = iter_fun(cfg, model, miner, idx, batch)
        rank_image_result.extend(batch_image_result)

    if WORLD_SIZE == 1:
        all_image_result = rank_image_result
    else:
        tmp_dir = './tmp_dir'
        all_image_result = collect_results_cpu(rank_image_result, num_image_all_rank, tmp_dir)

    if RANK in [0, -1]:
        mining_result = []
        for mining_info in all_image_result:
            mining_result.append((mining_info['image_filepath'], mining_info['score']))

        sorted_mining_result = sorted(mining_result, reverse=True, key=(lambda v: v[1]))
        with open(args.out_file, 'w') as fp:
            for img_path, score in sorted_mining_result:
                img_id = origin_image_list.index(img_path)
                fp.write(f"{img_id}\t{score}\n")
    return 0


if __name__ == '__main__':
    sys.exit(main())
