import pandas as pd
import numpy as np

import pytensor.tensor as pt

from sklearn.base import BaseEstimator, TransformerMixin

from typing import Union

class GeometricAdstockTransformer(BaseEstimator, TransformerMixin):
    """Transforms input data using the geometric adstock model.

    Args:
        alpha (float): The decay factor for the adstock transformation. Default is 0.0.
        l (int): The length of the adstock window. Default is 12.
    """

    def __init__(self, alpha: float = 0.0, l: int = 12):
        self.alpha = alpha
        self.l = l

    def fit(self, x: Union[pd.DataFrame, np.ndarray], y=None) -> "GeometricAdstockTransformer":
        """Fit the transformer to the input data.

        Args:
            x (pd.DataFrame or np.ndarray): The input data to fit the transformer on.
            y: Ignored. Present for compatibility.

        Returns:
            self (GeometricAdstockTransformer): The fitted transformer object.
        """
        return self

    def transform(self, x: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Transform the input data using the geometric adstock model.

        Args:
            x (pd.DataFrame or np.ndarray): The input data to transform.

        Returns:
            np.ndarray: The transformed data.
        """
        if isinstance(x, pd.DataFrame):
            x = x.to_numpy()
        cycles = [
            np.append(
                arr=np.zeros(shape=x.shape)[:i],
                values=x[: x.shape[0] - i],
                axis=0
            ) 
            for i in range(self.l)
        ]
        x_cycle = np.stack(cycles, axis=0)
        w = np.array([np.power(self.alpha, i) for i in range(self.l)])
        return np.tensordot(a=w, b=x_cycle, axes=1)
    

class LogisticSaturationTransformer(BaseEstimator, TransformerMixin):
    """Transforms input data using the logistic saturation model.

    Args:
        mu (float): The saturation factor for the transformation. Default is 0.5.
    """

    def __init__(self, mu: float = 0.5):
        self.mu = mu

    def fit(self, x: Union[pd.DataFrame, np.ndarray], y=None) -> "LogisticSaturationTransformer":
        """Fit the transformer to the input data.

        Args:
            x (pd.DataFrame or np.ndarray): The input data to fit the transformer on.
            y: Ignored. Present for compatibility.

        Returns:
            self (LogisticSaturationTransformer): The fitted transformer object.
        """
        return self

    def transform(self, x: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Transform the input data using the logistic saturation model.

        Args:
            x (pd.DataFrame or np.ndarray): The input data to transform.

        Returns:
            np.ndarray: The transformed data.
        """
        if isinstance(x, pd.DataFrame):
            x = x.to_numpy()
        return (1 - np.exp(-self.mu * x)) / (1 + np.exp(-self.mu * x))
    

def geometric_adstock(x, alpha: float = 0.0, l_max: int = 12):
    """Geometric adstock transformation."""
    cycles = [
        pt.concatenate(
            [pt.zeros(i), x[: x.shape[0] - i]]
        )
        for i in range(l_max)
    ]
    x_cycle = pt.stack(cycles)
    w = pt.as_tensor_variable([pt.power(alpha, i) for i in range(l_max)])
    return pt.dot(w, x_cycle)

def geometric_adstock_vectorized(x, alpha, l_max: int = 12, normalize: bool = False):
    """Vectorized geometric adstock transformation."""
    cycles = [
        pt.concatenate(tensor_list=[pt.zeros(shape=x.shape)[:i], x[: x.shape[0] - i]])
        for i in range(l_max)
    ]
    x_cycle = pt.stack(cycles)
    x_cycle = pt.transpose(x=x_cycle, axes=[1, 2, 0])
    w = pt.as_tensor_variable([pt.power(alpha, i) for i in range(l_max)])
    w = pt.transpose(w)[None, ...]

    if normalize:
        w = w / pt.sum(w)
        
    return pt.sum(pt.mul(x_cycle, w), axis=2)


def logistic_saturation(x, lam: float = 0.5):
    """Logistic saturation transformation."""
    return (1 - pt.exp(-lam * x)) / (1 + pt.exp(-lam * x))
