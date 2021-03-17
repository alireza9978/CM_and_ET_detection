from force_result.three import *
from optimizer.GrayWolfOptimizer import gray_wolf_optimizer

df = load_data_frame()

answer = gray_wolf_optimizer(calculate_accuracy, 0, 1, 5, 10, 3, True)

print(answer.convergence[-1])
print(answer.bestIndividual)

plt.plot(answer.convergence)
plt.show()
