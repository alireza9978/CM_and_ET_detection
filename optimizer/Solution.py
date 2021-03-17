
class Solution:
    def __init__(self):
        self.bestIndividual = []
        # todo add mean over time
        self.convergence = []
        self.optimizer = ""
        self.objective_function_name = ""
        self.startTime = 0
        self.endTime = 0
        self.executionTime = 0
        self.lb = 0
        self.ub = 0
        self.dim = 0
        self.popnum = 0
        self.maxiers = 0
