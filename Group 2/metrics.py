import numpy as np
from dataclasses import dataclass
from typing import Callable

@dataclass
class Metric:
	name: str
	threshold: str
	fn: Callable

def accuracy(y_pred, y_true):
    return (y_pred==y_true).mean()

def divergence_from_mean(X):
    #print( type(X) )
    #print( X.shape )
    #print( X.mean() )
    #print( X / X.mean() )
    return np.abs( X / X.mean() - 1 )

def changed_rate(y_new,y_old):
    return ( y_new != y_old ).mean()

ACCURACY_METRIC = Metric(
	name="Accuracy",
    threshold=0.8,
    fn = accuracy
)

DIVERGENCE_FROM_MEAN = Metric(
    name="Mean Div",
    threshold=0.05,
    fn = divergence_from_mean
)

CHANGE_RATE = Metric(
    name="Changed",
    threshold=0.05,
    fn = changed_rate
)
