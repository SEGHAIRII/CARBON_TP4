from abc import ABC, abstractmethod
import time
from Problem import FlowShopProblem

class AbstractOptimizer(ABC):
    def __init__(self, problem, **params):
        self.problem = problem
        self.params = params
        self.best_solution = None
        self.best_makespan = float('inf')
        self.execution_time = 0.0
        self.callback = None


    @abstractmethod
    def optimize(self):
       
        pass
    
    def set_callback(self, callback):
        self.callback = callback

    @classmethod
    @abstractmethod
    def suggest_params(cls, trial):
        """Hadi rah method ta optuna, n9dro nakhdmo b grid search ida habin"""
        pass


    def run(self):
       
        start_time = time.time()
        self.optimize()
        self.execution_time = time.time() - start_time

    def get_results(self):
      
        return {
            'schedule': self.best_solution,
            'makespan': self.best_makespan,
            'execution_time': self.execution_time
        }
