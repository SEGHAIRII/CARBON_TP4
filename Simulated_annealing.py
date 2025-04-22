import numpy as np
import random
import math
from Optimizer import AbstractOptimizer
from Problem import FlowShopProblem
import optuna
class SimulatedAnnealingOptimizer(AbstractOptimizer):

    def optimize(self):
        current_solution = self._neh_heuristic()
        current_makespan = self.problem.evaluate(current_solution)

        best_solution = current_solution.copy()
        best_makespan = current_makespan

        # Parameters
        T0 = self.params.get("initial_temperature", 1000)
        T = T0
        alpha = self.params.get("cooling_rate", 0.95)
        iterations = self.params.get("num_iterations", 1000)
        cooling_type = self.params.get("cooling_type", "continuous")
        cooling_method = self.params.get("cooling_method", "geometric")
        step_size = self.params.get("step_size", 50)
        stagnation_limit = self.params.get("stagnation_limit", 100)
        reheating_factor = self.params.get("reheating_factor", 0.1)

        last_improvement = 0
        no_improvement_count = 0

        for i in range(iterations):
            # Generate neighbor
            neighbor = current_solution.copy()
            a, b = random.sample(range(len(neighbor)), 2)
            neighbor[a], neighbor[b] = neighbor[b], neighbor[a]

            neighbor_makespan = self.problem.evaluate(neighbor)
            delta = neighbor_makespan - current_makespan

            if delta < 0 or random.random() < math.exp(-delta / T):
                current_solution = neighbor
                current_makespan = neighbor_makespan

                if current_makespan < best_makespan:
                    best_solution = current_solution.copy()
                    best_makespan = current_makespan
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
            else:
                no_improvement_count += 1

            # Update temperature
            T = self._update_temperature(T0, T, alpha, i, cooling_type, cooling_method, step_size, no_improvement_count, stagnation_limit, reheating_factor)

        self.best_solution = best_solution
        self.best_makespan = best_makespan

    def _update_temperature(self, T0, T, alpha, i, ctype, method, step, stagnation, stagnation_limit, reheating_factor):
        if ctype == "continuous":
            return self._continuous_cooling(T0, T, alpha, i, method)
        elif ctype == "stepwise":
            if i % step == 0 and i > 0:
                return self._continuous_cooling(T0, T, alpha, i, method)
            else:
                return T
        elif ctype == "non_monotonic":
            T_new = self._continuous_cooling(T0, T, alpha, i, method)
            if stagnation >= stagnation_limit:
                T_new *= (1 + reheating_factor)
            return T_new
        else:
            raise ValueError(f"Unknown cooling type: {ctype}")

    def _continuous_cooling(self, T0, T, alpha, iteration, method):
        if method == "geometric":
            return T * alpha
        elif method == "linear":
            return max(T - alpha, 1e-8)
        elif method == "exponential":
            return T0 * math.exp(-alpha * iteration)
        elif method == "logarithmic":
            return T0 / math.log(iteration + 2)
        else:
            raise ValueError(f"Unknown cooling method: {method}")

    @classmethod
    def suggest_params(cls, trial):
        return {
            "initial_temperature": trial.suggest_float("initial_temperature", 100, 5000),
            "cooling_rate": trial.suggest_float("cooling_rate", 0.001, 0.99),
            "num_iterations": trial.suggest_int("num_iterations", 500, 10000),
            "cooling_type": trial.suggest_categorical("cooling_type", ["continuous", "stepwise", "non_monotonic"]),
            "cooling_method": trial.suggest_categorical("cooling_method", ["geometric", "linear", "exponential", "logarithmic"]),
            "step_size": trial.suggest_int("step_size", 10, 200),
            "stagnation_limit": trial.suggest_int("stagnation_limit", 20, 300),
            "reheating_factor": trial.suggest_float("reheating_factor", 0.0, 0.5),
        }

    def _neh_heuristic(self):
        job_scores = [(j, np.sum(self.problem.processing_times[:, j])) for j in range(self.problem.num_jobs)]
        sorted_jobs = [job for job, _ in sorted(job_scores, key=lambda x: x[1], reverse=True)]

        partial_sequence = [sorted_jobs[0]]
        for job in sorted_jobs[1:]:
            best_seq = None
            best_makespan = float("inf")
            for i in range(len(partial_sequence) + 1):
                new_seq = partial_sequence[:i] + [job] + partial_sequence[i:]
                makespan = self.problem.evaluate(new_seq)
                if makespan < best_makespan:
                    best_makespan = makespan
                    best_seq = new_seq
            partial_sequence = best_seq
        return partial_sequence


if __name__ == "__main__":
    # Example usage

    problem = FlowShopProblem('./data/50_20_1.txt')
    study = optuna.create_study(direction='minimize')

    # Define the optimization loop
    def objective(trial):
        # Suggest parameters for the LocalSearchOptimizer
        params = SimulatedAnnealingOptimizer.suggest_params(trial)
        optimizer = SimulatedAnnealingOptimizer(problem, **params)
        optimizer.run()
        result = optimizer.get_results()
        print(result)
        return result['makespan']

    # Optimize the objective function with Optuna
    study.optimize(objective, n_trials=50)