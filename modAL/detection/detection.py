from typing import List

import numpy as np
from sklearn.base import BaseEstimator

from modAL.utils.selection import multi_argmax


def detection_uncertainty(detector: BaseEstimator, X: List[str], **predict_proba_kwargs) -> List[float]:
    bbox_results = detector.predict_proba(X, **predict_proba_kwargs)

    scores = [0.0 for i in range(len(X))]
    for idx, results in enumerate(bbox_results):
        if len(results) == 0:
            scores[idx] = 0
        elif len(results[0].boxes) == 0:
            scores[idx] = 0
        else:
            scores[idx] = 1 - float(np.max(results[0].boxes.conf.data.cpu().numpy()))

    return scores


def uncertainty_sampling(detector: BaseEstimator, X: List[str], n_instances: int = 1, **kwargs) -> List[int]:
    unc = detection_uncertainty(detector, X, **kwargs)
    indexes = multi_argmax(np.array(unc), n_instances=n_instances)
    return indexes.tolist()
