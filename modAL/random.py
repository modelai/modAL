import numpy as np
from sklearn.base import BaseEstimator

from modAL.utils.data import modALinput


def random_sampling(classifier: BaseEstimator,
                    X: modALinput,
                    n_instances: int = 1,
                    random_tie_break: bool = False,
                    **uncertainty_measure_kwargs) -> np.ndarray:
    return np.random.choice(range(len(X)), size=n_instances, replace=False)
