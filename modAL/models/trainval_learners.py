import abc
from typing import Any, Callable, List, Optional

from modAL.models.base import BaseLearner
from modAL.random import random_sampling
from modAL.utils.data import modALinput


class TrainValEstimator(object):

    @abc.abstractmethod
    def fit(self, X_train: List[str], y_train: List[str], X_val: List[str], y_val: List[str], **fit_kwargs) -> None:
        pass

    @abc.abstractmethod
    def score(self, X: Optional[List[str]] = None, y: Optional[List[str]] = None, **score_kwargs):
        pass

    @abc.abstractmethod
    def predict(self, X):
        pass

    @abc.abstractmethod
    def predict_proba(self, X):
        pass


class TrainValLearner(BaseLearner):
    """learning with train/val dataset

    keep the validation dataset unchanged, update training dataset.
    """

    def __init__(self,
                 estimator: TrainValEstimator,
                 X_val: modALinput,
                 y_val: modALinput,
                 query_strategy: Callable = random_sampling,
                 X_training: Optional[modALinput] = None,
                 y_training: Optional[modALinput] = None,
                 bootstrap_init: bool = False,
                 on_transformed: bool = False,
                 **fit_kwargs):

        self.X_val = X_val.copy()
        self.y_val = y_val.copy()

        super().__init__(estimator, query_strategy, X_training, y_training, bootstrap_init, on_transformed,
                         **fit_kwargs)

    def teach(self, X: modALinput, y: modALinput, **fit_kwargs) -> None:
        self._add_training_data(X, y)
        self.estimator.fit(self.X_training, self.y_training, self.X_val, self.y_val, **fit_kwargs)

    def _fit_to_known(self, bootstrap: bool = False, **fit_kwargs) -> 'BaseLearner':
        self.estimator.fit(self.X_training, self.y_training, self.X_val, self.y_val, **fit_kwargs)
        return self

    def _fit_on_new(self, X: modALinput, y: modALinput, bootstrap: bool = False, **fit_kwargs) -> 'BaseLearner':
        self.estimator.fit(X, y, self.X_val, self.y_val, **fit_kwargs)
        return self

    def fit(self, X: modALinput, y: modALinput, bootstrap: bool = False, **fit_kwargs) -> 'BaseLearner':
        self.X_training, self.y_training = X, y
        return self._fit_to_known(bootstrap=bootstrap, **fit_kwargs)

    def score(self, X: Optional[modALinput] = None, y: Optional[modALinput] = None, **score_kwargs) -> Any:
        return self.estimator.score(**score_kwargs)
