import random
import time
from Optimizer import AbstractOptimizer
from Problem import FlowShopProblem
import numpy as np

class Individual:
    def __init__(self, permutation, makespan):
        self.permutation = permutation
        self.makespan = makespan

class GeneticAlgorithmOptimizer(AbstractOptimizer):
    def __init__(self, problem, **params):
        super().__init__(problem, **params)
        # GA parameters
        self.population_size = params.get('population_size', 60)
        self.crossover_rate = params.get('crossover_rate', 0.8)
        self.mutation_rate = params.get('mutation_rate', 0.15)
        default_iters = 50 * (problem.get_num_jobs() + problem.get_num_machines())
        self.iterations = params.get('iterations', default_iters)
        self.seed = params.get('seed', None)
        # Selection settings
        self.selection_type = params.get('selection_type', 'roulette')  # 'roulette' or 'tournament'
        self.tournament_size = params.get('tournament_size', 3)
        # Crossover settings
        self.crossover_type = params.get('crossover_type', 'two_point')  # 'two_point' or 'one_point'
        # Mutation settings
        self.mutation_type = params.get('mutation_type', 'inversion')  # 'inversion' or 'swap'

    def _neh_sequence(self):
        proc_times = self.problem.get_processing_times()
        total_times = np.sum(proc_times, axis=0)
        jobs_sorted = list(map(int, np.argsort(-total_times)))
        sequence = [jobs_sorted[0]]
        for job in jobs_sorted[1:]:
            best_seq, best_ms = None, float('inf')
            for pos in range(len(sequence) + 1):
                trial = sequence[:pos] + [job] + sequence[pos:]
                ms = self.problem.evaluate(trial)
                if ms < best_ms:
                    best_ms, best_seq = ms, trial
            sequence = best_seq
        return sequence

    def optimize(self):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
        # Initial population
        population = []
        # NEH seed
        neh = self._neh_sequence()
        population.append(Individual(neh, self.problem.evaluate(neh)))
        base = list(range(self.problem.get_num_jobs()))
        for _ in range(self.population_size - 1):
            perm = base[:]
            random.shuffle(perm)
            population.append(Individual(perm, self.problem.evaluate(perm)))

        for _ in range(self.iterations):
            new_pop = []
            best = min(population, key=lambda i: i.makespan)
            new_pop.append(best)
            while len(new_pop) < self.population_size:
                # Selection
                if self.selection_type == 'tournament':
                    p1 = self._tournament_selection(population)
                    p2 = self._tournament_selection(population)
                else:
                    p1 = self._roulette_selection(population)
                    p2 = self._roulette_selection(population)
                # Crossover
                if random.random() < self.crossover_rate:
                    if self.crossover_type == 'one_point':
                        c1, c2 = self._one_point_crossover(p1, p2)
                    else:
                        c1, c2 = self._two_point_crossover(p1, p2)
                else:
                    c1 = Individual(p1.permutation[:], p1.makespan)
                    c2 = Individual(p2.permutation[:], p2.makespan)
                # Mutation
                for child in (c1, c2):
                    if random.random() < self.mutation_rate:
                        if self.mutation_type == 'swap':
                            self._swap_mutation(child)
                        else:
                            self._inversion_mutation(child)
                new_pop.extend([c1, c2][:self.population_size - len(new_pop)])
            population = new_pop

        best_final = min(population, key=lambda i: i.makespan)
        self.best_solution = [int(j) for j in best_final.permutation]
        self.best_makespan = best_final.makespan

    def _roulette_selection(self, pop):
        fits = [1.0 / ind.makespan for ind in pop]
        total = sum(fits)
        r = random.uniform(0, total)
        cum = 0
        for ind, f in zip(pop, fits):
            cum += f
            if cum >= r:
                return ind
        return pop[-1]

    def _tournament_selection(self, pop):
        cont = random.sample(pop, self.tournament_size)
        return min(cont, key=lambda i: i.makespan)

    def _one_point_crossover(self, p1, p2):
        n = len(p1.permutation)
        pt = random.randrange(1, n)
        seq1 = p1.permutation[:pt] + [j for j in p2.permutation if j not in p1.permutation[:pt]]
        seq2 = p2.permutation[:pt] + [j for j in p1.permutation if j not in p2.permutation[:pt]]
        return (Individual(seq1, self.problem.evaluate(seq1)),
                Individual(seq2, self.problem.evaluate(seq2)))

    def _two_point_crossover(self, p1, p2):
        n = len(p1.permutation)
        i, j = sorted(random.sample(range(n), 2))
        c1, c2 = p1.permutation[:], p2.permutation[:]
        c1[i:j+1], c2[i:j+1] = p2.permutation[i:j+1], p1.permutation[i:j+1]
        self._repair(c1); self._repair(c2)
        return (Individual(c1, self.problem.evaluate(c1)),
                Individual(c2, self.problem.evaluate(c2)))

    def _inversion_mutation(self, ind):
        perm = ind.permutation
        i, j = sorted(random.sample(range(len(perm)), 2))
        perm[i:j+1] = reversed(perm[i:j+1])
        ind.makespan = self.problem.evaluate(perm)

    def _swap_mutation(self, ind):
        perm = ind.permutation
        i, j = random.sample(range(len(perm)), 2)
        perm[i], perm[j] = perm[j], perm[i]
        ind.makespan = self.problem.evaluate(perm)

    def _repair(self, perm):
        n = len(perm); present = [False]*n
        for k in range(n):
            if 0 <= perm[k] < n and not present[perm[k]]:
                present[perm[k]] = True
            else: perm[k] = -1
        miss = 0
        for k in range(n):
            if perm[k] == -1:
                while present[miss]: miss += 1
                perm[k] = miss; present[miss] = True

    @classmethod
    def suggest_params(cls, trial):
        return {
            'population_size': trial.suggest_int('population_size', 20, 200),
            'crossover_rate': trial.suggest_float('crossover_rate', 0.5, 1.0),
            'mutation_rate': trial.suggest_float('mutation_rate', 0.0, 0.5),
            'iterations': trial.suggest_int('iterations', 100, 2000),
            'seed': trial.suggest_int('seed', 0, 10000),
            'selection_type': trial.suggest_categorical('selection_type', ['roulette', 'tournament']),
            'tournament_size': trial.suggest_int('tournament_size', 2, 10),
            'crossover_type': trial.suggest_categorical('crossover_type', ['two_point', 'one_point']),
            'mutation_type': trial.suggest_categorical('mutation_type', ['inversion', 'swap'])
        }

if __name__ == "__main__":
    problem = FlowShopProblem('./data/20_20_1.txt')
    params = {
        'population_size': 30,
        'crossover_rate': 0.8,
        'mutation_rate': 0.15,
        'iterations': 200,
        'seed': 42,
        'selection_type': 'roulette',
        'tournament_size': 3,
        'crossover_type': 'two_point',
        'mutation_type': 'inversion'
    }
    optimizer = GeneticAlgorithmOptimizer(problem, **params)
    optimizer.run()
    results = optimizer.get_results()
    print(f"Best makespan: {results['makespan']}")
    print(f"Best schedule: {results['schedule']}")
    print(f"Execution time: {results['execution_time']:.4f}s")

    # # Uncomment to use suggest_params with Optuna
    # import optuna
    # study = optuna.create_study(direction='minimize')
    # def objective(trial):
    #     suggested = GeneticAlgorithmOptimizer.suggest_params(trial)
    #     opt = GeneticAlgorithmOptimizer(problem, **suggested)
    #     opt.run()
    #     return opt.get_results()['makespan']
    # study.optimize(objective, n_trials=50)
    # print(f"Best hyperparameters: {study.best_params}")
    # print(f"Best makespan: {study.best_value}")
