import os
import subprocess
import warnings

import numpy as np
from modAL.utils.data import modALinput
from sklearn.base import BaseEstimator


def ripu_sampling(classifier: BaseEstimator,
                  X: modALinput,
                  n_instances: int = 1,
                  omega_config_file: str = './omega_config.yaml',
                  **uncertainty_measure_kwargs) -> np.ndarray:

    # need abspath
    for img_path in X:
        if os.path.abspath(img_path) != img_path:
            warnings.warn(f'the image path {img_path} in X is not abosolute path')

    index_file = './candidate-index.txt'
    with open(index_file, 'w') as fp:
        for img_path in X:
            fp.write(f'{img_path}\n')

    out_file = './mining-result.txt'
    cmd = f'python3 modAL/segmentation/ripu.py --omega_config {omega_config_file}'
    cmd += f' --index_file {index_file} --out_file {out_file}'

    subprocess.run(cmd.split(), check=True)

    with open(out_file, 'r') as fr:
        query_index = []
        for line in fr.readlines():
            query_index.append(int(line.strip().split()[0]))

    return np.array(query_index[0:n_instances])


def rsal_sampling(classifier: BaseEstimator,
                  X: modALinput,
                  n_instances: int = 1,
                  omega_config_file: str = './omega_config.yaml',
                  **uncertainty_measure_kwargs) -> np.ndarray:

    # need abspath
    for img_path in X:
        if os.path.abspath(img_path) != img_path:
            warnings.warn(f'the image path {img_path} in X is not abosolute path')

    index_file = './candidate-index.txt'
    with open(index_file, 'w') as fp:
        for img_path in X:
            fp.write(f'{img_path}\n')

    out_file = './mining-result.txt'
    cmd = f'python3 modAL/segmentation/rsal.py --omega_config {omega_config_file}'
    cmd += f' --index_file {index_file} --out_file {out_file}'

    subprocess.run(cmd.split(), check=True)

    with open(out_file, 'r') as fr:
        query_index = []
        for line in fr.readlines():
            query_index.append(int(line.strip().split()[0]))

    return np.array(query_index[0:n_instances])
