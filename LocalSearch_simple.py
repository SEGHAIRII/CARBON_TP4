import numpy as np
import random
from scipy.spatial.distance import euclidean
from Optimizer import AbstractOptimizer
import optuna
from Problem import FlowShopProblem


class LocalSearchOptimizer(AbstractOptimizer):
    def __init__(self, problem, **params):
        super().__init__(problem, **params)
        # Set local search parameters
        self.neighborhood_size = params.get('neighborhood_size', 10)  # Number of neighbors to explore
        self.step_size = params.get('step_size', 1)  # Perturbation size for swapping jobs

    def optimize(self):
        # Start with a random permutation of jobs
        current_solution = list(np.random.permutation(self.problem.num_jobs))
        # Evaluate the initial solution and extract only the makespan (first value of the tuple)
        current_makespan = self.problem.evaluate(current_solution)

        # Explore the neighborhood for a maximum of `neighborhood_size` iterations
        for _ in range(self.neighborhood_size):
            best_neighbor = current_solution
            best_makespan = current_makespan

            # Generate neighbors by swapping two jobs in the current solution
            for i in range(self.problem.num_jobs):
                for j in range(i + 1, self.problem.num_jobs):
                    neighbor = current_solution[:]
                    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]  # Swap
                    # Evaluate the neighbor and extract the makespan
                    makespan = self.problem.evaluate(neighbor)

                    if makespan < best_makespan:
                        best_makespan = makespan
                        best_neighbor = neighbor

            # If a better neighbor is found, update the current solution
            if best_makespan < current_makespan:
                current_solution = best_neighbor
                current_makespan = best_makespan

        # Store the best solution found
        self.best_solution = current_solution
        self.best_makespan = current_makespan


    @classmethod
    def suggest_params(cls, trial):
        """
        Suggest parameters using Optuna. This includes the number of
        iterations (neighborhood size) and perturbation step size.
        """
        return {
            'neighborhood_size': trial.suggest_int('neighborhood_size', 5, 50),
            'step_size': trial.suggest_int('step_size', 1, 5),  # Can be used for more complex perturbation
        }
        
        
        
if __name__ == "__main__":

    # Load the problem
    problem = FlowShopProblem('./data/20_5_1.txt')

    # Create an Optuna study to minimize makespan
    study = optuna.create_study(direction='minimize')

    # Define the optimization loop
    def objective(trial):
        # Suggest parameters for the LocalSearchOptimizer
        params = LocalSearchOptimizer.suggest_params(trial)
        optimizer = LocalSearchOptimizer(problem, **params)
        optimizer.run()
        result = optimizer.get_results()
        print(result)
        return result['makespan']

    # Optimize the objective function with Optuna
    study.optimize(objective, n_trials=50)

    # Print the best hyperparameters and result
    print(f"Best Hyperparameters: {study.best_params}")
    print(f"Best Makespan: {study.best_value}")
