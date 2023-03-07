"""
RSAL: Revisiting Superpixels for Active Learning in Semantic Segmentation With Realistic Annotation Costs (CVPR 2021)

view code https://github.com/cailile/Revisiting-Superpixels-for-Active-Learning
"""
import argparse
import logging
import os
import os.path as osp
import sys
import warnings
from typing import Dict, List, Union

import cv2
import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.engine import collect_results_cpu
from mmcv.runner import init_dist, wrap_fp16_model
from modAL.segmentation.superpixel import get_superpixel
from modAL.segmentation.utils import get_dataloader
from omegaconf import DictConfig, ListConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def get_uncertainty_info(cfg: Union[DictConfig, ListConfig], result: torch.Tensor,
                         superpixel: np.ndarray) -> List[Dict]:
    """
    input:
        result: prediction result with softmax propability for each class
        superpixel: superpixel labels

    return:
        unc_list: uncertainty information for each superpixel region
            superpixel_idx: index
            score: uncertainty score
            class_id: class id for each superpixel region if need class balance
    """
    uncertainty_method: str = cfg.mining.uncertainty_method
    need_class_balance: bool = cfg.mining.class_balance

    c, h, w = result.shape
    sph, spw = superpixel.shape

    assert h == sph and w == spw, f'result shape {h, w} != superpixel shape {sph, spw}'

    if uncertainty_method == 'BvSB':
        # the bigger the better
        if c == 1:
            uncertainty = 1 - torch.abs(result - 0.5)
        else:
            # torch.sort is faster than torch.topk when c is small (<100)
            sorted_result, _ = torch.sort(result, dim=0, descending=True)
            uncertainty = sorted_result[1, :, :] / sorted_result[0, :, :]
    else:
        raise Exception(f'unknown uncertainty method {uncertainty_method}')

    np_unc = uncertainty.data.cpu().numpy()
    Max = superpixel.max()
    if need_class_balance:
        pred = torch.argmax(result, dim=0).data.cpu().numpy()

    unc_list: List[Dict] = []
    for idx in range(Max):
        y, x = np.where(superpixel == idx)

        # remove empty superpixel
        if len(x) == 0:
            warnings.warn(f'remove empty superpixel {idx}')
            continue

        score = float(np.mean(np_unc[y, x]))
        d = dict(superpixel_idx=idx, score=score)
        if need_class_balance:
            unique, unique_counts = np.unique(pred[y, x], return_counts=True)
            class_id = unique[np.argmax(unique_counts)]
            d['class_id'] = class_id

        unc_list.append(d)
    return unc_list


def iter_fun(cfg, model, idx, batch) -> List[Dict]:
    """
    batch: Dict(img=[tensor,], img_metas=[List[DataContainer],])

    return: image filename list and correspond topk scores
        image_results: List[Dict]
            image_filepath: image file path
            superpixel_filepath: superpixel label file path
            unc_list: uncertainty info for superpixel
    """
    # result = inference_segmentor(model, image_filename)[0]
    batch_img = batch['img'][0]
    batch_meta = batch['img_metas'][0].data[0]
    device = torch.device('cuda:0' if RANK == -1 else f'cuda:{RANK}')
    batch_result = model.inference(batch_img.to(device), batch_meta, rescale=True)

    image_results = []
    save_dir = cfg.output.root_dir
    os.makedirs(osp.join(save_dir, 'superpixel'), exist_ok=True)
    max_superpixel_per_image = int(cfg.mining.max_superpixel_per_image)
    for idx, meta in enumerate(batch_meta):

        png_basename = osp.splitext(osp.basename(meta['filename']))[0] + '.png'
        superpixel_filepath = osp.join(save_dir, 'superpixel', png_basename)
        if osp.exists(superpixel_filepath):
            load_img = Image.open(superpixel_filepath, mode='I')
            superpixel = np.array(load_img)
        else:
            superpixel = get_superpixel(cfg, img=meta['filename'], max_superpixels=max_superpixel_per_image)
            # cannot use cv2.imwrite to save superpixel
            pil_img = Image.fromarray(superpixel, mode='I')

            pil_img.save(superpixel_filepath)

        unc_list = get_uncertainty_info(cfg, batch_result[idx], superpixel)

        image_results.append(
            dict(image_filepath=meta['filename'], superpixel_filepath=superpixel_filepath, unc_list=unc_list))
    return image_results


def update_image_scores(region_scores: List[Dict], threshold: float, topk: int):
    for dict_per_img in tqdm(region_scores, desc='update image scores'):
        unc_list = sorted(dict_per_img['unc_list'], key=lambda a: a['score'], reverse=True)
        topk_scores = []
        image_score = 0
        for idx, d in enumerate(unc_list):
            if idx < topk:
                # image scores = sum of topk region scores
                image_score += d['score']
                topk_scores.append(d)
            elif d['score'] >= threshold:
                # allow label more than topk region better than threshold
                image_score += d['score']
                topk_scores.append(d)
            else:
                break

        dict_per_img['image_score'] = image_score
        dict_per_img['topk_scores'] = topk_scores


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
        batch_image_result = iter_fun(cfg, model, idx, batch)
        rank_image_result.extend(batch_image_result)

    if WORLD_SIZE == 1:
        all_image_result = rank_image_result
    else:
        tmp_dir = 'tmp_dir'
        all_image_result = collect_results_cpu(rank_image_result, num_image_all_rank, tmp_dir)

    if RANK in [0, -1]:
        # note we remove normalization here
        need_class_balance: bool = cfg.mining.class_balance
        if need_class_balance:
            # add background class
            class_num = len(cfg.data.class_names)
            region_num_per_class = np.zeros(shape=(class_num), dtype=np.float32)
            for unc_info in tqdm(all_image_result, desc='apply class weight'):
                for d in unc_info['unc_list']:
                    region_num_per_class[d['class_id']] += 1

            total_region_num = np.sum(region_num_per_class)
            class_weights = np.exp(-region_num_per_class / total_region_num)

        all_region_scores = []
        for img_idx, unc_info in enumerate(tqdm(all_image_result, desc='gather all region scores')):
            for d in unc_info['unc_list']:
                if need_class_balance:
                    d['score'] = d['score'] * class_weights[d['class_id']]
                all_region_scores.append(d['score'])

        # 80% scores <= threshold
        max_kept_mining_image = int(cfg.mining.max_kept_mining_image)
        topk_superpixel_score = int(cfg.mining.topk_superpixel_score)

        # make sure topk * num_images superpixel be labeled.
        percent = round(100 * max(0.8, 1 - max_kept_mining_image * topk_superpixel_score / len(all_region_scores)))
        threshold = np.percentile(all_region_scores, percent)
        logging.info(f'region scores: len={len(all_region_scores)}, percentile-{percent}={threshold}')
        logging.info(f'region scores: max={np.max(all_region_scores)}, min={np.min(all_region_scores)}')

        # add image_score and sort the image result by score
        update_image_scores(all_image_result, threshold, topk_superpixel_score)
        all_image_result = sorted(all_image_result, key=lambda x: x['image_score'], reverse=True)

        # create mask to label and remove superpixel file
        os.makedirs(osp.join(cfg.output.root_dir, 'masks'), exist_ok=True)
        mining_result = []
        for img_idx in range(num_image_all_rank):
            if img_idx < max_kept_mining_image:
                superpixel = np.array(Image.open(all_image_result[img_idx]['superpixel_filepath']))
                topk_index = [d['superpixel_idx'] for d in all_image_result[img_idx]['topk_scores']]
                mask_to_label = np.zeros(superpixel.shape, dtype=np.bool8)
                for superpixel_idx in topk_index:
                    mask_to_label = np.logical_or(mask_to_label, superpixel == superpixel_idx)

                mask_name = osp.join(cfg.output.root_dir, 'masks',
                                     osp.basename(all_image_result[img_idx]['image_filepath']))
                cv2.imwrite(mask_name, 255 * mask_to_label.astype(np.uint8))

            mining_result.append(
                (all_image_result[img_idx]['image_filepath'], all_image_result[img_idx]['image_score']))
            os.remove(all_image_result[img_idx]['superpixel_filepath'])

        sorted_mining_result = sorted(mining_result, reverse=True, key=(lambda v: v[1]))
        with open(args.out_file, 'w') as fp:
            for img_path, score in sorted_mining_result:
                img_id = origin_image_list.index(img_path)
                fp.write(f"{img_id}\t{score}\n")
    return 0


if __name__ == '__main__':
    sys.exit(main())
