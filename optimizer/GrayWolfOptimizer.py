import random
import time
from typing import Callable

import numpy as np

from optimizer.Solution import Solution


def gray_wolf_optimizer(objective_function: Callable[[np.array], int], lower_band, upper_band, dimension: int,
                        search_agents_number: int = 10, max_iteration: int = 100,
                        enable_print: bool = False) -> Solution:
    # initialize alpha, beta, and delta_pos
    alpha_pos = np.zeros(dimension)
    alpha_score = float("inf")

    beta_pos = np.zeros(dimension)
    beta_score = float("inf")

    delta_pos = np.zeros(dimension)
    delta_score = float("inf")

    if not isinstance(lower_band, list):
        lower_band = [lower_band] * dimension
    if not isinstance(upper_band, list):
        upper_band = [upper_band] * dimension

    # Initialize the positions of search agents
    positions = np.zeros((search_agents_number, dimension))
    for i in range(dimension):
        positions[:, i] = (
                np.random.uniform(0, 1, search_agents_number) * (upper_band[i] - lower_band[i]) + lower_band[i])

    convergence_curve = np.zeros(max_iteration)
    s = Solution()

    if enable_print:
        # Loop counter
        print('GWO is optimizing  "' + objective_function.__name__ + '"')

    timer_start = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    # Main loop
    for iteration_number in range(max_iteration):
        for i in range(search_agents_number):
            # Return back the search agents that go beyond the boundaries of the search space
            for j in range(dimension):
                positions[i, j] = np.clip(positions[i, j], lower_band[j], upper_band[j])

            # Calculate objective function for each search agent
            fitness = objective_function(positions[i, :])

            # Update Alpha, Beta, and Delta
            if fitness < alpha_score:
                delta_score = beta_score  # Update delta
                delta_pos = beta_pos.copy()
                beta_score = alpha_score  # Update beta
                beta_pos = alpha_pos.copy()
                alpha_score = fitness
                # Update alpha
                alpha_pos = positions[i, :].copy()

            if alpha_score < fitness < beta_score:
                delta_score = beta_score  # Update delta
                delta_pos = beta_pos.copy()
                beta_score = fitness  # Update beta
                beta_pos = positions[i, :].copy()

            if alpha_score < fitness < delta_score and fitness > beta_score:
                delta_score = fitness  # Update delta
                delta_pos = positions[i, :].copy()

        # a decreases linearly from 2 to 0
        a = 2 - iteration_number * (2 / max_iteration)

        # Update the Position of search agents including omegas
        for i in range(0, search_agents_number):
            for j in range(0, dimension):
                r1 = random.random()  # r1 is a random number in [0,1]
                r2 = random.random()  # r2 is a random number in [0,1]

                a1 = 2 * a * r1 - a
                # Equation (3.3)
                c1 = 2 * r2
                # Equation (3.4)

                d_alpha = abs(c1 * alpha_pos[j] - positions[i, j])
                # Equation (3.5)-part 1
                x1 = alpha_pos[j] - a1 * d_alpha
                # Equation (3.6)-part 1

                r1 = random.random()
                r2 = random.random()

                a2 = 2 * a * r1 - a
                # Equation (3.3)
                c2 = 2 * r2
                # Equation (3.4)

                d_beta = abs(c2 * beta_pos[j] - positions[i, j])
                # Equation (3.5)-part 2
                x2 = beta_pos[j] - a2 * d_beta
                # Equation (3.6)-part 2

                r1 = random.random()
                r2 = random.random()

                a3 = 2 * a * r1 - a
                # Equation (3.3)
                c3 = 2 * r2
                # Equation (3.4)

                d_delta = abs(c3 * delta_pos[j] - positions[i, j])
                # Equation (3.5)-part 3
                x3 = delta_pos[j] - a3 * d_delta
                # Equation (3.5)-part 3

                positions[i, j] = (x1 + x2 + x3) / 3  # Equation (3.7)

        convergence_curve[iteration_number] = alpha_score

        if enable_print:
            if iteration_number % 5 == 0:
                print(["At iteration_number " + str(iteration_number) + " the best fitness is " + str(alpha_score)])

    timer_end = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.bestIndividual = alpha_pos
    s.executionTime = timer_end - timer_start
    s.convergence = convergence_curve
    s.optimizer = "GWO"
    s.objective_function_name = objective_function.__name__

    return s
