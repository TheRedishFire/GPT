import numpy as np
from numpy.typing import NDArray
  
def sigmoid(z: NDArray[np.float64]) -> NDArray[np.float64]:
    sig = 1 / (1 + np.exp(-z))
    return np.round(sig, 5)


def relu(z: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.maximum(z, 0)
