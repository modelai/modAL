import json
from pprint import pprint

import numpy as np
from modAL.datasets.cityscapes import CityscapesDataset
from modAL.models.mmseg import MMSegEstimator
from modAL.models.trainval_learners import TrainValLearner
from modAL.random import random_sampling
from modAL.segmentation.mining import ripu_sampling, rsal_sampling
from omegaconf import OmegaConf

np.random.seed(25)
config = OmegaConf.load('configs/mmseg/mmseg_cityscapes.yaml')

dataset = {}
X = {}
y = {}

for split in ['train', 'val']:
    dataset[split] = CityscapesDataset(config, split)

    X[split] = dataset[split].img_paths
    y[split] = dataset[split].lbl_paths

    if split == 'val':
        sample_idx = np.random.choice(range(len(X[split])), size=500, replace=False)
        X[split] = [X[split][idx] for idx in sample_idx]
        y[split] = [y[split][idx] for idx in sample_idx]

# assemble initial data
n_initial = 100
initial_idx = np.random.choice(range(len(X['train'])), size=n_initial, replace=False)

X_initial = [X['train'][idx] for idx in initial_idx]
y_initial = [y['train'][idx] for idx in initial_idx]

# generate the pool
# remove the initial data from the training dataset
pool_index = [idx for idx in range(len(X['train'])) if idx not in initial_idx]
X_pool = [X['train'][idx] for idx in pool_index]
y_pool = [y['train'][idx] for idx in pool_index]

model = MMSegEstimator(config)
mining_algorithm = config.mining.mining_algorithm
if mining_algorithm == 'random':
    query_strategy = random_sampling
elif mining_algorithm == 'ripu':
    query_strategy = ripu_sampling  # type: ignore
elif mining_algorithm == 'rsal':
    query_strategy = rsal_sampling  # type: ignore
else:
    assert False

learner = TrainValLearner(estimator=model, X_val=X['val'], y_val=y['val'], query_strategy=query_strategy)

# init model
learner.fit(X_initial, y_initial, max_epoches=10, project_name=f'{mining_algorithm}_init')
init_score = learner.score()

# the active learning loop
trained_idx = initial_idx.copy()
n_queries = 10
max_epoches = [50, 50, 40, 40, 30, 30, 20, 20, 10, 10]
max_epoches = [i // 10 for i in max_epoches]

scores = dict(random_init=init_score)

for idx in range(n_queries):
    query_idx, query_instance = learner.query(X_pool, n_instances=100, omega_config_file=model.dump_omega_config())

    origin_query_idx = [pool_index[idx] for idx in query_idx]
    X_query = [X['train'][idx] for idx in origin_query_idx]
    y_query = [y['train'][idx] for idx in origin_query_idx]
    learner.teach(X_query,
                  y_query,
                  only_new=False,
                  max_epoches=max_epoches[idx],
                  project_name=f'{mining_algorithm}_{idx}')
    # remove queried instance from pool
    trained_idx = np.concatenate([origin_query_idx, trained_idx])
    pool_index = [idx for idx in range(len(X['train'])) if idx not in trained_idx]
    X_pool = [X['train'][idx] for idx in pool_index]
    y_pool = [y['train'][idx] for idx in pool_index]

    # the final accuracy score
    score = learner.score()
    scores[f'{mining_algorithm}_{idx}'] = score
    print(f'query {idx}, score = {score}')

print('the active learning scores are: ')
pprint(scores)

with open(f'{mining_algorithm}_result.yaml', 'w') as fp:
    json.dump(scores, fp)
