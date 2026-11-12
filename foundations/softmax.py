import numpy as np
from numpy.typing import NDArray
def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
    shifted = z - np.max(z)
    exps = np.exp(shifted)
    result = exps / np.sum(exps)
    return np.round(result, 4)