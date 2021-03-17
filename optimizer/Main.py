import matplotlib.pyplot as plt
import numpy as np

from optimizer.GrayWolfOptimizer import gray_wolf_optimizer


def test_function(temp: np.array) -> int:
    return


answer = gray_wolf_optimizer(test_function, -10, 10, 5, 10, 100)

print(answer.convergence[-1])

plt.plot(answer.convergence)
plt.show()
