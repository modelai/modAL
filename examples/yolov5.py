from pprint import pprint

import numpy as np
from modAL.datasets.voc import VOCDataset
from modAL.detection.detection import uncertainty_sampling
from modAL.models import ActiveLearner
from modAL.models.yolov5 import Yolov5Estimator
from omegaconf import OmegaConf

config = OmegaConf.load('configs/yolov5/yolov5_voc.yaml')

train_dataset = VOCDataset(config, split='train')
val_dataset = VOCDataset(config, split='val')
test_dataset = VOCDataset(config, split='test')

X_train, y_train = train_dataset.img_paths, train_dataset.lbl_paths
X_val, y_val = val_dataset.img_paths, val_dataset.lbl_paths
X_test, y_test = test_dataset.img_paths, test_dataset.lbl_paths

# assemble initial data
n_initial = 1000
initial_idx = np.random.choice(range(len(X_train)), size=n_initial, replace=False)

X_initial = [X_train[idx] for idx in initial_idx]
y_initial = [y_train[idx] for idx in initial_idx]

# generate the pool
# remove the initial data from the training dataset
pool_index = [idx for idx in range(len(X_train)) if idx not in initial_idx]
X_pool = [X_train[idx] for idx in pool_index]
y_pool = [y_train[idx] for idx in pool_index]

# will use the val dataset to evaluation training result
detector = Yolov5Estimator(config)
learner = ActiveLearner(estimator=detector,
                        query_strategy=uncertainty_sampling,
                        X_training=X_initial,
                        y_training=y_initial)
init_score = learner.score(X_test, y_test)

# the active learning loop
trained_idx = initial_idx.copy()
n_queries = 10
scores = [init_score]

for idx in range(n_queries):
    query_idx, query_instance = learner.query(X_pool, n_instances=100)

    origin_query_idx = [pool_index[idx] for idx in query_idx]
    X_query = [X_train[idx] for idx in origin_query_idx]
    y_query = [y_train[idx] for idx in origin_query_idx]
    learner.teach(X_query, y_query, only_new=False)
    # remove queried instance from pool
    trained_idx = origin_query_idx + trained_idx
    pool_index = [idx for idx in range(len(X_train)) if idx not in trained_idx]
    X_pool = [X_train[idx] for idx in pool_index]
    y_pool = [y_train[idx] for idx in pool_index]

    # the final accuracy score
    score = learner.score(X_test, y_test)
    scores.append(score)
    print(f'query {idx}, score = {score}')

print('the active learning scores are: ')
pprint(scores)
