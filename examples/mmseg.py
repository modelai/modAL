from pprint import pprint

import numpy as np
from modAL.datasets.cityscapes import CityscapesDataset
from modAL.models.mmseg import MMSegEstimator
from modAL.models.trainval_learners import TrainValLearner
from omegaconf import OmegaConf

config = OmegaConf.load('configs/mmseg/mmseg_cityscapes.yaml')

dataset = {}
X = {}
y = {}
for split in ['train', 'val']:
    dataset[split] = CityscapesDataset(config, split)

    X[split] = dataset[split].img_paths
    y[split] = dataset[split].lbl_paths

# assemble initial data
n_initial = 1000
initial_idx = np.random.choice(range(len(X['train'])), size=n_initial, replace=False)

X_initial = [X['train'][idx] for idx in initial_idx]
y_initial = [y['train'][idx] for idx in initial_idx]

# generate the pool
# remove the initial data from the training dataset
pool_index = [idx for idx in range(len(X['train'])) if idx not in initial_idx]
X_pool = [X['train'][idx] for idx in pool_index]
y_pool = [y['train'][idx] for idx in pool_index]

model = MMSegEstimator(config)
learner = TrainValLearner(estimator=model, X_val=X['val'], y_val=y['val'])

# init model
learner.fit(X_initial, y_initial)
init_score = learner.score()

# the active learning loop
# trained_idx = initial_idx.copy()
# n_queries = 10
# scores = [init_score]

# for idx in range(n_queries):
#     query_idx, query_instance = learner.query(X_pool, n_instances=100)

#     origin_query_idx = [pool_index[idx] for idx in query_idx]
#     X_query = [X['train'][idx] for idx in origin_query_idx]
#     y_query = [y['train'][idx] for idx in origin_query_idx]
#     learner.teach(X_query, y_query, only_new=False)
#     # remove queried instance from pool
#     trained_idx = origin_query_idx + trained_idx
#     pool_index = [idx for idx in range(len(X['train'])) if idx not in trained_idx]
#     X_pool = [X['train'][idx] for idx in pool_index]
#     y_pool = [y['train'][idx] for idx in pool_index]

#     # the final accuracy score
#     score = learner.score()
#     scores.append(score)
#     print(f'query {idx}, score = {score}')

# print('the active learning scores are: ')
# pprint(scores)
