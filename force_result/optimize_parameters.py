from geneticalgorithm import geneticalgorithm as ga

from force_result.three import *

varbound = np.array([[0, 7 * 24], [0, 50], [0, 4], [0, 30], [0.5, 1]])
vartype = np.array([['int'], ['int'], ['real'], ['int'], ["real"]])
algorithm_param = {'max_num_iteration': 10,
                   'population_size': 5,
                   'mutation_probability': 0.1,
                   'elit_ratio': 0.01,
                   'crossover_probability': 0.5,
                   'parents_portion': 0.3,
                   'crossover_type': 'uniform',
                   'max_iteration_without_improv': None}

model = ga(function=calculate_accuracy, dimension=5, variable_type_mixed=vartype, variable_boundaries=varbound,
           function_timeout=1000, algorithm_parameters=algorithm_param)
model.run()

# answer = gray_wolf_optimizer(calculate_accuracy, 0, 1, 5, 10, 3, True)
#
# print(answer.convergence[-1])
# print(answer.bestIndividual)
#
# plt.plot(answer.convergence)
# plt.show()
