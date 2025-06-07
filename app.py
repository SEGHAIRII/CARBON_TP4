from flask import Flask, render_template, request, redirect, url_for, jsonify, Response
import os
import time
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
import numpy as np
import threading
import csv
from flask_socketio import SocketIO
import traceback
import math
import random

# Import our optimization algorithms
from Problem import FlowShopProblem
from AntSystem import AntSystemOptimizer
from Genetic import GeneticAlgorithmOptimizer, Individual  # Import Individual class
from LocalSearch_simple import LocalSearchOptimizer
from Simulated_annealing import SimulatedAnnealingOptimizer
from iterated_beam_search import Iterated_Beam_Search

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

app = Flask(__name__)
app.json_encoder = NumpyEncoder
socketio = SocketIO(app, cors_allowed_origins="*", json=json)

# Store for optimization results
results_history = []
# Store for ongoing optimizations
ongoing_optimizations = {}
# Store for algorithm descriptions
algorithm_descriptions = {
    "ant_system": {
        "name": "Ant System",
        "description": "A population-based metaheuristic inspired by the foraging behavior of ants. It uses pheromone trails to guide the search process.",
        "strengths": ["Good for complex problems", "Avoids local optima", "Parallelizable"],
        "weaknesses": ["Parameter tuning can be difficult", "Can be computationally intensive"],
        "parameters": {
            "alpha": "Controls the influence of pheromone trails (higher values give more weight to pheromone)",
            "beta": "Controls the influence of heuristic information (higher values give more weight to shorter paths)",
            "visibility_strat": "Strategy for calculating visibility (local or total makespan)",
            "q": "Pheromone intensity factor",
            "ro": "Pheromone evaporation rate (0-1)",
            "m": "Number of ants in the colony",
            "sigma0": "Initial pheromone value",
            "n": "Number of iterations",
            "e": "Elitism factor (0 = no elitism, higher values increase elitism)"
        }
    },
    "iterated_beam_search": {
        "name": "Iterated Beam Search",
        "description": "A tree based search method that uses multiple iterations of beam search with different beam sizes and second chance pruning strategy",
        "strengths": ["Good for complex problems", "Avoids local optima", "Parallelizable"],
        "weaknesses": ["not much parameter tuning", "Can be computationally intensive"],
        "parameters": {
            "max iterations": "Maximum number of iterations",
            "initial beam width": "the beam width used in the first beam search",
            "beam width factor": "factor used to multiply each beam size with from iteration to the next",
            "time limit": "maximum execution time allowed",
        }
    },
    "genetic": {
        "name": "Genetic Algorithm",
        "description": "An evolutionary algorithm inspired by natural selection. It evolves a population of solutions through selection, crossover, and mutation.",
        "strengths": ["Good for complex search spaces", "Can find good solutions quickly", "Highly parallelizable"],
        "weaknesses": ["Parameter tuning can be difficult", "No guarantee of optimality"],
        "parameters": {
            "population_size": "Number of individuals in the population",
            "iterations": "Number of generations to evolve",
            "crossover_rate": "Probability of crossover (0-1)",
            "mutation_rate": "Probability of mutation (0-1)",
            "selection_type": "Method for selecting parents (tournament or roulette wheel)",
            "tournament_size": "Number of individuals in tournament selection",
            "crossover_type": "Method for crossover (one-point or two-point)",
            "mutation_type": "Method for mutation (swap or inversion)",
            "seed": "Random seed for reproducibility"
        }
    },
    "local_search": {
        "name": "Local Search",
        "description": "A simple hill-climbing algorithm that iteratively improves a solution by exploring its neighborhood.",
        "strengths": ["Simple to implement", "Fast for small problems", "Low memory requirements"],
        "weaknesses": ["Can get stuck in local optima", "Not suitable for complex problems"],
        "parameters": {
            "neighborhood_size": "Number of neighbors to explore",
            "step_size": "Size of perturbation when generating neighbors"
        }
    },
    "simulated_annealing": {
        "name": "Simulated Annealing",
        "description": "A probabilistic technique inspired by the annealing process in metallurgy. It allows accepting worse solutions with a decreasing probability to escape local optima.",
        "strengths": ["Can escape local optima", "Works well for complex problems", "Relatively simple to implement"],
        "weaknesses": ["Parameter tuning can be difficult", "Convergence can be slow"],
        "parameters": {
            "initial_temperature": "Starting temperature (higher = more exploration)",
            "cooling_rate": "Rate at which temperature decreases",
            "num_iterations": "Number of iterations to run",
            "cooling_type": "Method for temperature reduction (continuous, stepwise, non-monotonic)",
            "cooling_method": "Formula for temperature reduction (geometric, linear, exponential, logarithmic)",
            "step_size": "Number of iterations before temperature reduction in stepwise cooling",
            "stagnation_limit": "Number of iterations without improvement before reheating",
            "reheating_factor": "Factor by which to increase temperature when reheating"
        }
    }
}

class ProgressTracker:
    def __init__(self, optimizer_id, total_iterations):
        self.optimizer_id = optimizer_id
        self.total_iterations = total_iterations
        self.current_iteration = 0
        self.best_makespan = float('inf')
        self.start_time = time.time()
        self.status = "running"
        self.message = "Initializing..."
        
    def update(self, iteration, makespan):
        self.current_iteration = iteration
        if makespan < self.best_makespan:
            self.best_makespan = makespan
        self.message = f"Iteration {iteration}/{self.total_iterations}, Best makespan: {self.best_makespan}"
        print(type(iteration))
        print(type(self.total_iterations))
        progress = min(100, int((iteration / self.total_iterations) * 100))
        
        # Emit progress update via Socket.IO
        socketio.emit('optimization_progress', {
            'id': self.optimizer_id,
            'progress': progress,
            'current_iteration': iteration,
            'total_iterations': self.total_iterations,
            'best_makespan': float(self.best_makespan),  # Convert NumPy types to native Python
            'elapsed_time': time.time() - self.start_time,
            'status': self.status,
            'message': self.message
        })
        
    def complete(self, makespan, solution):
        self.status = "completed"
        self.best_makespan = makespan
        self.message = f"Optimization completed. Best makespan: {makespan}"
        
        # Convert NumPy types to native Python
        if isinstance(makespan, np.number):
            makespan = float(makespan)
        
        if isinstance(solution, np.ndarray):
            solution = solution.tolist()
        else:
            # Convert any NumPy integers in the list to Python integers
            solution = [int(x) if isinstance(x, np.integer) else x for x in solution]
            
        socketio.emit('optimization_complete', {
            'id': self.optimizer_id,
            'best_makespan': makespan,
            'solution': solution,
            'elapsed_time': time.time() - self.start_time,
            'status': self.status,
            'message': self.message
        })
        
    def error(self, message):
        self.status = "error"
        self.message = f"Error: {message}"
        socketio.emit('optimization_error', {
            'id': self.optimizer_id,
            'message': message,
            'status': self.status
        })

# Custom optimizer wrappers to track progress
class TrackableAntSystemOptimizer(AntSystemOptimizer):
    def __init__(self, problem, tracker, **params):
        super().__init__(problem, **params)
        self.tracker = tracker
        
    def optimize(self):
        # Override the optimize method to track progress
        current_solution = list(np.random.permutation(self.problem.num_jobs))
        current_makespan = self.problem.evaluate(current_solution)

        self.alpha = self.params.get('alpha', 1.0)
        self.beta = self.params.get('beta', 2.0)
        self.visibility_strat = self.params.get('visibility_strat', 'local_makespan')
        self.q = 50*(self.problem.num_machines+self.problem.num_jobs)*self.params.get('q', 1.0)
        self.ro = self.params.get('ro', 0.5)
        self.m = self.params.get('m', 44)
        self.sigma0 = self.params.get('sigma0', 0.2)
        self.n = self.params.get('n', 500)
        self.e = self.params.get('e', 1.0)

        self.pheromoneGraph = np.full((self.problem.num_jobs, self.problem.num_jobs), self.sigma0)
        self.frames = []
        
        # Before starting the construction process, we need a reference solution
        start_time = time.time()
        frames = [self.pheromoneGraph]
        current_solution = list(np.random.permutation(self.problem.num_jobs))
        current_makespan = self.problem.evaluate(current_solution)
        
        functions = {
            "total_makespan": self.total_makespan,
            "local_makespan": self.local_makespan
        }
        visibility = functions[self.visibility_strat]
        
        for iteration in range(self.n):
            # Update progress tracker
            self.tracker.update(iteration + 1, current_makespan)
            
            nb_jobs = self.problem.num_jobs
            deltaPheromon = np.zeros((self.problem.num_jobs, self.problem.num_jobs))
            average_makespan = 0
            ants_log = []
            
            for ant in range(self.m):
                available_jobs = list(range(nb_jobs))
                path = []
                first_step = np.random.randint(0, nb_jobs)
                path.append(first_step)
                available_jobs.remove(first_step)
                
                while len(available_jobs) > 1:
                    current_job = path[-1]
                    score_list = [(self.pheromoneGraph[current_job, job]**self.alpha) * 
                                 ((1/visibility(job, path))**self.beta) for job in available_jobs]
                    total = sum(score_list)
                    score_list = np.array(score_list)
                    distribution = (score_list+(0.001/len(available_jobs))) / (total+0.001)
                    sampled_job_index = np.random.choice(len(distribution), p=distribution)
                    selected_job = available_jobs[sampled_job_index]
                    path.append(selected_job)
                    available_jobs.remove(selected_job)
                
                path.append(available_jobs[0])
                path_makespan = self.problem.evaluate(path)
                
                if path_makespan < current_makespan:
                    current_solution = path
                    current_makespan = path_makespan
                
                average_makespan += path_makespan
                deltaSigma = self.q / path_makespan
                
                for arc in range(len(path)-1):
                    deltaPheromon[path[arc], path[arc+1]] += deltaSigma
                
                ants_log.append((path, path_makespan))
            
            best_iteration_path, best_iteration_path_makespan = min(ants_log, key=lambda x: x[1])
            
            # Elitism
            for arc in range(len(best_iteration_path)-1):
                deltaPheromon[best_iteration_path[arc], best_iteration_path[arc+1]] += self.e * self.q / best_iteration_path_makespan
            
            self.pheromoneGraph = self.pheromoneGraph * (1-self.ro) + deltaPheromon
            frames.append(self.pheromoneGraph)
        
        self.best_makespan = current_makespan
        self.best_solution = current_solution
        end_time = time.time()
        self.frames = frames
        self.execution_time = end_time - start_time
        
        # Mark optimization as complete
        self.tracker.complete(self.best_makespan, self.best_solution)

class TrackableGeneticAlgorithmOptimizer(GeneticAlgorithmOptimizer):
    def __init__(self, problem, tracker, **params):
        super().__init__(problem, **params)
        self.tracker = tracker
        
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

        best = min(population, key=lambda i: i.makespan)
        self.tracker.update(0, best.makespan)

        for iteration in range(self.iterations):
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
            best = min(population, key=lambda i: i.makespan)
            self.tracker.update(iteration + 1, best.makespan)

        best_final = min(population, key=lambda i: i.makespan)
        self.best_solution = [int(j) for j in best_final.permutation]
        self.best_makespan = best_final.makespan
        
        # Mark optimization as complete
        self.tracker.complete(self.best_makespan, self.best_solution)

class TrackableLocalSearchOptimizer(LocalSearchOptimizer):
    def __init__(self, problem, tracker, **params):
        super().__init__(problem, **params)
        self.tracker = tracker
        
    def optimize(self):
        # Start with a random permutation of jobs
        current_solution = list(np.random.permutation(self.problem.num_jobs))
        current_makespan = self.problem.evaluate(current_solution)
        
        self.tracker.update(0, current_makespan)
        
        # Explore the neighborhood for a maximum of `neighborhood_size` iterations
        for iteration in range(self.neighborhood_size):
            best_neighbor = current_solution
            best_makespan = current_makespan

            # Generate neighbors by swapping two jobs in the current solution
            for i in range(self.problem.num_jobs):
                for j in range(i + 1, self.problem.num_jobs):
                    neighbor = current_solution[:]
                    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]  # Swap
                    makespan = self.problem.evaluate(neighbor)

                    if makespan < best_makespan:
                        best_makespan = makespan
                        best_neighbor = neighbor

            # If a better neighbor is found, update the current solution
            if best_makespan < current_makespan:
                current_solution = best_neighbor
                current_makespan = best_makespan
                
            self.tracker.update(iteration + 1, current_makespan)

        # Store the best solution found
        self.best_solution = [int(x) for x in current_solution]  # Convert to regular Python integers
        self.best_makespan = float(current_makespan)  # Convert to regular Python float
        
        # Mark optimization as complete
        self.tracker.complete(self.best_makespan, self.best_solution)

class TrackableSimulatedAnnealingOptimizer(SimulatedAnnealingOptimizer):
    def __init__(self, problem, tracker, **params):
        super().__init__(problem, **params)
        self.tracker = tracker
        
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
        
        self.tracker.update(0, best_makespan)

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
            
            # Update progress every 10 iterations to avoid overwhelming the UI
            if i % 10 == 0 or i == iterations - 1:
                self.tracker.update(i + 1, best_makespan)

        # Convert NumPy types to native Python types
        self.best_solution = [int(x) for x in best_solution]
        self.best_makespan = float(best_makespan)
        
        # Mark optimization as complete
        self.tracker.complete(self.best_makespan, self.best_solution)


    

@app.route('/')
def index():
    # Get list of available problem instances
    problem_instances = sorted([f for f in os.listdir('data') if f.endswith('.txt')])
    
    # Define available algorithms
    algorithms = [
        {"id": "ant_system", "name": "Ant System"},
        {"id": "genetic", "name": "Genetic Algorithm"},
        {"id": "local_search", "name": "Local Search"},
        {"id": "simulated_annealing", "name": "Simulated Annealing"},
        {"id": "iterated_beam_search", "name": "Iterated Beam Search"},
        
        
    ]
    
    return render_template('index.html', problem_instances=problem_instances, algorithms=algorithms, algorithm_descriptions=algorithm_descriptions)

@app.route('/get_algorithm_params', methods=['POST'])
def get_algorithm_params():
    algorithm = request.json.get('algorithm')
    
    # Return the appropriate parameter form based on the selected algorithm
    if algorithm == 'ant_system':
        return jsonify({
            'params': [
                {'id': 'alpha', 'name': 'Alpha (Pheromone Influence)', 'type': 'number', 'default': 1.0, 'min': 0.1, 'max': 5.0, 'step': 0.1, 
                 'description': algorithm_descriptions['ant_system']['parameters']['alpha']},
                {'id': 'beta', 'name': 'Beta (Visibility Influence)', 'type': 'number', 'default': 2.0, 'min': 0.1, 'max': 5.0, 'step': 0.1,
                 'description': algorithm_descriptions['ant_system']['parameters']['beta']},
                {'id': 'visibility_strat', 'name': 'Visibility Strategy', 'type': 'select', 
                 'options': [{'value': 'total_makespan', 'text': 'Total Makespan'}, {'value': 'local_makespan', 'text': 'Local Makespan'}],
                 'default': 'total_makespan',
                 'description': algorithm_descriptions['ant_system']['parameters']['visibility_strat']},
                {'id': 'q', 'name': 'Q (Pheromone Intensity)', 'type': 'number', 'default': 2.0, 'min': 0.1, 'max': 10.0, 'step': 0.1,
                 'description': algorithm_descriptions['ant_system']['parameters']['q']},
                {'id': 'ro', 'name': 'Rho (Evaporation Rate)', 'type': 'number', 'default': 0.5, 'min': 0.1, 'max': 0.9, 'step': 0.1,
                 'description': algorithm_descriptions['ant_system']['parameters']['ro']},
                {'id': 'm', 'name': 'Number of Ants', 'type': 'number', 'default': 20, 'min': 5, 'max': 100, 'step': 1,
                 'description': algorithm_descriptions['ant_system']['parameters']['m']},
                {'id': 'sigma0', 'name': 'Initial Pheromone', 'type': 'number', 'default': 0.1, 'min': 0.01, 'max': 1.0, 'step': 0.01,
                 'description': algorithm_descriptions['ant_system']['parameters']['sigma0']},
                {'id': 'n', 'name': 'Number of Iterations', 'type': 'number', 'default': 100, 'min': 10, 'max': 1000, 'step': 10,
                 'description': algorithm_descriptions['ant_system']['parameters']['n']},
                {'id': 'e', 'name': 'Elitism Factor', 'type': 'number', 'default': 1.0, 'min': 0.0, 'max': 5.0, 'step': 0.1,
                 'description': algorithm_descriptions['ant_system']['parameters']['e']},
            ]
        })
    elif algorithm == 'genetic':
        return jsonify({
            'params': [
                {'id': 'population_size', 'name': 'Population Size', 'type': 'number', 'default': 100, 'min': 10, 'max': 500, 'step': 10,
                 'description': algorithm_descriptions['genetic']['parameters']['population_size']},
                {'id': 'iterations', 'name': 'Number of Generations', 'type': 'number', 'default': 100, 'min': 10, 'max': 1000, 'step': 10,
                 'description': algorithm_descriptions['genetic']['parameters']['iterations']},
                {'id': 'crossover_rate', 'name': 'Crossover Rate', 'type': 'number', 'default': 0.8, 'min': 0.1, 'max': 1.0, 'step': 0.05,
                 'description': algorithm_descriptions['genetic']['parameters']['crossover_rate']},
                {'id': 'mutation_rate', 'name': 'Mutation Rate', 'type': 'number', 'default': 0.2, 'min': 0.01, 'max': 0.5, 'step': 0.01,
                 'description': algorithm_descriptions['genetic']['parameters']['mutation_rate']},
                {'id': 'selection_type', 'name': 'Selection Method', 'type': 'select', 
                 'options': [{'value': 'tournament', 'text': 'Tournament'}, {'value': 'roulette', 'text': 'Roulette Wheel'}],
                 'default': 'tournament',
                 'description': algorithm_descriptions['genetic']['parameters']['selection_type']},
                {'id': 'tournament_size', 'name': 'Tournament Size', 'type': 'number', 'default': 3, 'min': 2, 'max': 10, 'step': 1,
                 'description': algorithm_descriptions['genetic']['parameters']['tournament_size']},
                {'id': 'crossover_type', 'name': 'Crossover Type', 'type': 'select',
                 'options': [{'value': 'two_point', 'text': 'Two Point'}, {'value': 'one_point', 'text': 'One Point'}],
                 'default': 'two_point',
                 'description': algorithm_descriptions['genetic']['parameters']['crossover_type']},
                {'id': 'mutation_type', 'name': 'Mutation Type', 'type': 'select',
                 'options': [{'value': 'inversion', 'text': 'Inversion'}, {'value': 'swap', 'text': 'Swap'}],
                 'default': 'inversion',
                 'description': algorithm_descriptions['genetic']['parameters']['mutation_type']},
                {'id': 'seed', 'name': 'Random Seed', 'type': 'number', 'default': 42, 'min': 0, 'max': 10000, 'step': 1,
                 'description': algorithm_descriptions['genetic']['parameters']['seed']},
            ]
        })
    elif algorithm == 'local_search':
        return jsonify({
            'params': [
                {'id': 'neighborhood_size', 'name': 'Neighborhood Size', 'type': 'number', 'default': 10, 'min': 5, 'max': 100, 'step': 5,
                 'description': algorithm_descriptions['local_search']['parameters']['neighborhood_size']},
                {'id': 'step_size', 'name': 'Step Size', 'type': 'number', 'default': 1, 'min': 1, 'max': 5, 'step': 1,
                 'description': algorithm_descriptions['local_search']['parameters']['step_size']},
            ]
        })
    elif algorithm == 'simulated_annealing':
        return jsonify({
            'params': [
                {'id': 'initial_temperature', 'name': 'Initial Temperature', 'type': 'number', 'default': 1000.0, 'min': 100.0, 'max': 5000.0, 'step': 100.0,
                 'description': algorithm_descriptions['simulated_annealing']['parameters']['initial_temperature']},
                {'id': 'cooling_rate', 'name': 'Cooling Rate', 'type': 'number', 'default': 0.95, 'min': 0.001, 'max': 0.99, 'step': 0.01,
                 'description': algorithm_descriptions['simulated_annealing']['parameters']['cooling_rate']},
                {'id': 'num_iterations', 'name': 'Number of Iterations', 'type': 'number', 'default': 1000, 'min': 500, 'max': 10000, 'step': 100,
                 'description': algorithm_descriptions['simulated_annealing']['parameters']['num_iterations']},
                {'id': 'cooling_type', 'name': 'Cooling Type', 'type': 'select',
                 'options': [
                     {'value': 'continuous', 'text': 'Continuous'},
                     {'value': 'stepwise', 'text': 'Stepwise'},
                     {'value': 'non_monotonic', 'text': 'Non-Monotonic'}
                 ],
                 'default': 'continuous',
                 'description': algorithm_descriptions['simulated_annealing']['parameters']['cooling_type']},
                {'id': 'cooling_method', 'name': 'Cooling Method', 'type': 'select',
                 'options': [
                     {'value': 'geometric', 'text': 'Geometric'},
                     {'value': 'linear', 'text': 'Linear'},
                     {'value': 'exponential', 'text': 'Exponential'},
                     {'value': 'logarithmic', 'text': 'Logarithmic'}
                 ],
                 'default': 'geometric',
                 'description': algorithm_descriptions['simulated_annealing']['parameters']['cooling_method']},
                {'id': 'step_size', 'name': 'Step Size', 'type': 'number', 'default': 50, 'min': 10, 'max': 200, 'step': 10,
                 'description': algorithm_descriptions['simulated_annealing']['parameters']['step_size']},
                {'id': 'stagnation_limit', 'name': 'Stagnation Limit', 'type': 'number', 'default': 100, 'min': 20, 'max': 300, 'step': 10,
                 'description': algorithm_descriptions['simulated_annealing']['parameters']['stagnation_limit']},
                {'id': 'reheating_factor', 'name': 'Reheating Factor', 'type': 'number', 'default': 0.1, 'min': 0.0, 'max': 0.5, 'step': 0.05,
                 'description': algorithm_descriptions['simulated_annealing']['parameters']['reheating_factor']},
            ]
        })
    elif algorithm == 'iterated_beam_search':
        return jsonify({
            'params': [
                {'id': 'initial_beam_width', 'name': 'Initial Beam Width', 'type': 'number', 'default': 1, 'min': 1, 'max': 100, 'step': 1,
                 'description': algorithm_descriptions['iterated_beam_search']['parameters']['initial beam width']},
                {'id': 'max_ibs_iterations', 'name': 'Max Iterations', 'type': 'number', 'default': 5, 'min': 1, 'max': 100, 'step': 1,
                 'description': algorithm_descriptions['iterated_beam_search']['parameters']['max iterations']},
                {'id': 'beam_width_factor', 'name': 'Beam Width Factor', 'type': 'number', 'default': 2, 'min': 1, 'max': 100, 'step': 1,
                 'description': algorithm_descriptions['iterated_beam_search']['parameters']['beam width factor']},
                {'id': 'time_limit_seconds', 'name': 'Time Limit', 'type': 'number', 'default': None, 'min': 1, 'max': 10000, 'step': 1,
                 'description': algorithm_descriptions['iterated_beam_search']['parameters']['time limit']},   
            ]
        })
    else:
        return jsonify({'params': []})
    
@app.route('/get_problem_instances')
def get_problem_instances():
    problem_instances = [f for f in os.listdir('data') if f.endswith('.txt')]
    return jsonify(sorted(problem_instances))

@app.route('/run_optimization', methods=['POST'])
def run_optimization():
    data = request.json
    problem_instance = data.get('problem_instance')
    algorithm = data.get('algorithm')
    params = data.get('params', {})
    
    # Generate a unique ID for this optimization
    optimizer_id = f"{algorithm}_{problem_instance}_{int(time.time())}"
    
    # Load the problem
    problem_path = os.path.join('data', problem_instance)
    problem = FlowShopProblem(problem_path)
    
    # Convert parameters to appropriate types
    for key, value in params.items():
        if isinstance(value, str):
            # Convert string values to appropriate types
            if key in ['alpha', 'beta', 'q', 'ro', 'sigma0', 'e', 'crossover_rate', 'mutation_rate', 
                      'initial_temperature', 'cooling_rate', 'reheating_factor']:
                params[key] = float(value)
            elif key in ['m', 'n', 'population_size', 'iterations', 'num_iterations', 'tournament_size', 
                        'neighborhood_size', 'step_size', 'stagnation_limit', 'seed']:
                params[key] = int(value)
            elif key == 'first_improvement':
                params[key] = value.lower() == 'true'
    
    # Determine the total iterations for progress tracking
    total_iterations = 100  # Default
    if algorithm == 'ant_system':
        total_iterations = params.get('n', 100)
    elif algorithm == 'genetic':
        total_iterations = params.get('iterations', 100)
    elif algorithm == 'local_search':
        total_iterations = params.get('neighborhood_size', 10)
    elif algorithm == 'simulated_annealing':
        total_iterations = params.get('num_iterations', 1000)
    elif algorithm == 'iterated_beam_search':
        total_iterations = int(params.get('max_ibs_iterations', 10))
    
    
    tracker = ProgressTracker(optimizer_id, total_iterations)
    ongoing_optimizations[optimizer_id] = tracker
    
    # Start the optimization in a separate thread
    def run_optimization_thread():
        try:
            if algorithm == 'ant_system':
                optimizer = TrackableAntSystemOptimizer(problem, tracker, **params)
            elif algorithm == 'genetic':
                optimizer = TrackableGeneticAlgorithmOptimizer(problem, tracker, **params)
            elif algorithm == 'local_search':
                optimizer = TrackableLocalSearchOptimizer(problem, tracker, **params)
            elif algorithm == 'simulated_annealing':
                optimizer = TrackableSimulatedAnnealingOptimizer(problem, tracker, **params)
            elif algorithm == 'iterated_beam_search':
                optimizer = Iterated_Beam_Search(problem, tracker, **params)
            else:
                tracker.error('Invalid algorithm selection')
                return
            
            # Run optimization
            optimizer.optimize()
            
            # Extract results
            result = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'problem_instance': problem_instance,
                'algorithm': algorithm,
                'params': params,
                'makespan': float(optimizer.best_makespan) if isinstance(optimizer.best_makespan, np.number) else optimizer.best_makespan,
                'execution_time': optimizer.execution_time if hasattr(optimizer, 'execution_time') else time.time() - tracker.start_time,
                'solution': [int(x) if isinstance(x, np.integer) else x for x in optimizer.best_solution]
            }
            
            # Save to history
            results_history.append(result)
            
            # If we have too many results, keep only the last 20
            if len(results_history) > 20:
                results_history.pop(0)
            
            # Generate solution visualization for algorithms that support it
            solution_viz = None
            if algorithm == 'ant_system' and hasattr(optimizer, 'frames') and len(optimizer.frames) > 0:
                # For ant system, we can plot pheromone matrix evolution
                plt.figure(figsize=(8, 6))
                plt.imshow(optimizer.frames[-1], cmap='hot', interpolation='nearest')
                plt.colorbar(label='Pheromone Intensity')
                plt.title('Final Pheromone Matrix')
                plt.tight_layout()
                
                # Convert plot to base64 string
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                solution_viz = base64.b64encode(buf.read()).decode('utf-8')
                plt.close()
            
            # Generate Gantt chart for best solution
            gantt_chart = generate_gantt_chart(problem, optimizer.best_solution)
            
            # Clean up
            if optimizer_id in ongoing_optimizations:
                del ongoing_optimizations[optimizer_id]
            
            # Emit final result
            socketio.emit('optimization_result', {
                'id': optimizer_id,
                'makespan': float(optimizer.best_makespan) if isinstance(optimizer.best_makespan, np.number) else optimizer.best_makespan,
                'execution_time': float(result['execution_time']),
                'solution': result['solution'],
                'solution_viz': solution_viz,
                'gantt_chart': gantt_chart
            })
            
        except Exception as e:
            traceback.print_exc()
            tracker.error(str(e))
            if optimizer_id in ongoing_optimizations:
                del ongoing_optimizations[optimizer_id]
    
    # Start the thread
    thread = threading.Thread(target=run_optimization_thread)
    thread.daemon = True
    thread.start()
    
    # Return the optimizer ID so the client can track progress
    return jsonify({
        'optimizer_id': optimizer_id,
        'status': 'started'
    })

@app.route('/optimization_status/<optimizer_id>', methods=['GET'])
def optimization_status(optimizer_id):
    if optimizer_id in ongoing_optimizations:
        tracker = ongoing_optimizations[optimizer_id]
        return jsonify({
            'id': optimizer_id,
            'progress': min(100, int((tracker.current_iteration / tracker.total_iterations) * 100)),
            'current_iteration': tracker.current_iteration,
            'total_iterations': tracker.total_iterations,
            'best_makespan': float(tracker.best_makespan) if isinstance(tracker.best_makespan, np.number) else tracker.best_makespan,
            'elapsed_time': time.time() - tracker.start_time,
            'status': tracker.status,
            'message': tracker.message
        })
    else:
        return jsonify({
            'id': optimizer_id,
            'status': 'not_found',
            'message': 'Optimization not found'
        })

@app.route('/cancel_optimization/<optimizer_id>', methods=['POST'])
def cancel_optimization(optimizer_id):
    if optimizer_id in ongoing_optimizations:
        tracker = ongoing_optimizations[optimizer_id]
        tracker.status = 'cancelled'
        tracker.message = 'Optimization cancelled by user'
        del ongoing_optimizations[optimizer_id]
        return jsonify({'status': 'cancelled'})
    else:
        return jsonify({'status': 'not_found'})

@app.route('/history')
def history():
    return render_template('history.html', results=results_history)

@app.route('/compare')
def compare():
    return render_template('compare.html')

@app.route('/get_history', methods=['GET'])
def get_history():
    # Convert NumPy types to native Python types
    serializable_history = []
    for result in results_history:
        serializable_result = {
            'timestamp': result['timestamp'],
            'problem_instance': result['problem_instance'],
            'algorithm': result['algorithm'],
            'params': result['params'],
            'makespan': float(result['makespan']) if isinstance(result['makespan'], np.number) else result['makespan'],
            'execution_time': float(result['execution_time']),
            'solution': [int(x) if isinstance(x, np.integer) else x for x in result['solution']]
        }
        serializable_history.append(serializable_result)
    
    return jsonify(serializable_history)

@app.route('/export_history', methods=['GET'])
def export_history():
    format_type = request.args.get('format', 'csv')
    
    if format_type == 'json':
        # Convert any NumPy types to native Python types
        serializable_history = []
        for result in results_history:
            serializable_result = {
                'timestamp': result['timestamp'],
                'problem_instance': result['problem_instance'],
                'algorithm': result['algorithm'],
                'params': result['params'],
                'makespan': float(result['makespan']) if isinstance(result['makespan'], np.number) else result['makespan'],
                'execution_time': float(result['execution_time']),
                'solution': [int(x) if isinstance(x, np.integer) else x for x in result['solution']]
            }
            serializable_history.append(serializable_result)
        
        return jsonify(serializable_history)
    elif format_type == 'csv':
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(['Timestamp', 'Problem Instance', 'Algorithm', 'Makespan', 'Execution Time', 'Parameters', 'Solution'])
        
        # Write data
        for result in results_history:
            writer.writerow([
                result['timestamp'],
                result['problem_instance'],
                result['algorithm'],
                float(result['makespan']) if isinstance(result['makespan'], np.number) else result['makespan'],
                float(result['execution_time']),
                json.dumps(result['params']),
                ','.join(str(int(x) if isinstance(x, np.integer) else x) for x in result['solution'])
            ])
        
        output.seek(0)
        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-disposition": "attachment; filename=flowshop_history.csv"}
        )
    else:
        return jsonify({'error': 'Invalid format type'})

@app.route('/clear_history', methods=['POST'])
def clear_history():
    global results_history
    results_history = []
    return jsonify({'status': 'success', 'message': 'History cleared successfully'})

@app.route('/algorithm_info/<algorithm_id>', methods=['GET'])
def algorithm_info(algorithm_id):
    if algorithm_id in algorithm_descriptions:
        return jsonify(algorithm_descriptions[algorithm_id])
    else:
        return jsonify({'error': 'Algorithm not found'})

def generate_gantt_chart(problem, solution):
    """Generate a Gantt chart for the given solution"""
    jobs = len(solution)
    machines = problem.num_machines
    
    # Calculate completion times
    completion_times = np.zeros((jobs, machines))
    
    for job_idx, job in enumerate(solution):
        for machine in range(machines):
            if job_idx == 0 and machine == 0:
                completion_times[job_idx][machine] = problem.processing_times[machine][job]
            elif job_idx == 0:
                completion_times[job_idx][machine] = completion_times[job_idx][machine-1] + problem.processing_times[machine][job]
            elif machine == 0:
                completion_times[job_idx][machine] = completion_times[job_idx-1][machine] + problem.processing_times[machine][job]
            else:
                completion_times[job_idx][machine] = max(completion_times[job_idx-1][machine], 
                                                       completion_times[job_idx][machine-1]) + problem.processing_times[machine][job]
    
    # Create Gantt chart
    plt.figure(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, jobs))
    
    for job_idx, job in enumerate(solution):
        for machine in range(machines):
            if job_idx == 0 and machine == 0:
                start_time = 0
            elif job_idx == 0:
                start_time = completion_times[job_idx][machine-1] - problem.processing_times[machine][job]
            elif machine == 0:
                start_time = completion_times[job_idx-1][machine]
            else:
                start_time = max(completion_times[job_idx-1][machine], 
                                completion_times[job_idx][machine-1])
            
            duration = problem.processing_times[machine][job]
            plt.barh(machine, duration, left=start_time, height=0.5, 
                    color=colors[job_idx], alpha=0.8)
            
            # Add job number label in the middle of each bar
            plt.text(start_time + duration/2, machine, f'J{job}', 
                    ha='center', va='center', color='black', fontweight='bold')
    
    plt.yticks(range(machines), [f'M{i}' for i in range(machines)])
    plt.xlabel('Time')
    plt.ylabel('Machine')
    plt.title('Gantt Chart for Best Solution')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    # Convert plot to base64 string
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    gantt_chart = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return gantt_chart

@app.route('/compare_algorithms', methods=['POST'])
def compare_algorithms():
    data = request.json
    problem_instance = data.get('problem_instance')
    algorithms = data.get('algorithms', [])
    
    if not problem_instance or not algorithms:
        return jsonify({'error': 'Missing required parameters'})
    
    # Load the problem
    problem_path = os.path.join('data', problem_instance)
    problem = FlowShopProblem(problem_path)
    
    results = []
    
    for algorithm_data in algorithms:
        algorithm = algorithm_data.get('algorithm')
        params = algorithm_data.get('params', {})
        
        # Convert parameters to appropriate types
        for key, value in params.items():
            if isinstance(value, str):
                if key in ['alpha', 'beta', 'q', 'ro', 'sigma0', 'e', 'crossover_rate', 'mutation_rate', 
                          'initial_temperature', 'cooling_rate', 'reheating_factor']:
                    params[key] = float(value)
                elif key in ['m', 'n', 'population_size', 'iterations', 'num_iterations', 'tournament_size', 
                            'neighborhood_size', 'step_size', 'stagnation_limit', 'seed']:
                    params[key] = int(value)
                elif key == 'first_improvement':
                    params[key] = value.lower() == 'true'
        
        # Run the algorithm
        start_time = time.time()
        try:
            if algorithm == 'ant_system':
                optimizer = AntSystemOptimizer(problem, **params)
            elif algorithm == 'genetic':
                optimizer = GeneticAlgorithmOptimizer(problem, **params)
            elif algorithm == 'local_search':
                optimizer = LocalSearchOptimizer(problem, **params)
            elif algorithm == 'simulated_annealing':
                optimizer = SimulatedAnnealingOptimizer(problem, **params)
            else:
                return jsonify({'error': f'Invalid algorithm selection: {algorithm}'})
            
            # Run optimization
            optimizer.optimize()
            execution_time = time.time() - start_time
            
            # Extract results
            result = {
                'algorithm': algorithm,
                'algorithm_name': algorithm_descriptions[algorithm]['name'],
                'params': params,
                'makespan': float(optimizer.best_makespan) if isinstance(optimizer.best_makespan, np.number) else optimizer.best_makespan,
                'execution_time': float(execution_time),
                'solution': [int(x) if isinstance(x, np.integer) else x for x in optimizer.best_solution]
            }
            
            results.append(result)
            
        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': f'Error running {algorithm}: {str(e)}'})
    
    # Sort results by makespan
    results.sort(key=lambda x: x['makespan'])
    
    # Generate comparison chart
    plt.figure(figsize=(10, 6))
    algorithms = [r['algorithm_name'] for r in results]
    makespans = [r['makespan'] for r in results]
    times = [r['execution_time'] for r in results]
    
    x = np.arange(len(algorithms))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot makespan bars
    bars1 = ax1.bar(x - width/2, makespans, width, label='Makespan', color='steelblue')
    ax1.set_ylabel('Makespan', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    
    # Create a second y-axis for execution time
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, times, width, label='Execution Time (s)', color='coral')
    ax2.set_ylabel('Execution Time (s)', color='coral')
    ax2.tick_params(axis='y', labelcolor='coral')
    
    # Add labels and title
    ax1.set_xlabel('Algorithm')
    ax1.set_title(f'Algorithm Comparison for {problem_instance}')
    ax1.set_xticks(x)
    ax1.set_xticklabels(algorithms, rotation=45, ha='right')
    
    # Add a legend
    ax1.legend(handles=[bars1, bars2], loc='upper left')
    
    plt.tight_layout()
    
    # Convert plot to base64 string
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    comparison_chart = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return jsonify({
        'results': results,
        'comparison_chart': comparison_chart
    })

@app.route('/get_problem_info/<problem_instance>', methods=['GET'])
def get_problem_info(problem_instance):
    problem_path = os.path.join('data', problem_instance)
    
    try:
        problem = FlowShopProblem(problem_path)
        
        # Generate a visualization of the processing times
        plt.figure(figsize=(10, 6))
        plt.imshow(problem.processing_times, cmap='viridis', aspect='auto')
        plt.colorbar(label='Processing Time')
        plt.xlabel('Job')
        plt.ylabel('Machine')
        plt.title(f'Processing Times for {problem_instance}')
        
        # Add text annotations for processing times
        for i in range(problem.num_machines):
            for j in range(problem.num_jobs):
                plt.text(j, i, str(problem.processing_times[i, j]),
                        ha="center", va="center", color="white" if problem.processing_times[i, j] > 10 else "black")
        
        plt.tight_layout()
        
        # Convert plot to base64 string
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        processing_times_viz = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return jsonify({
            'num_jobs': int(problem.num_jobs),
            'num_machines': int(problem.num_machines),
            'processing_times': problem.processing_times.tolist(),
            'processing_times_viz': processing_times_viz
        })
    except Exception as e:
        return jsonify({'error': str(e)})

# Add this new route for algorithm recommendation
@app.route('/recommend_algorithm', methods=['POST'])
def recommend_algorithm():
    data = request.json
    problem_instance = data.get('problem_instance')
    
    if not problem_instance:
        return jsonify({'error': 'Missing problem instance'})
    
    # Load the problem
    problem_path = os.path.join('data', problem_instance)
    try:
        problem = FlowShopProblem(problem_path)
        
        # Analyze problem characteristics
        num_jobs = problem.num_jobs
        num_machines = problem.num_machines
        processing_times = problem.processing_times
        
        # Calculate some metrics
        avg_processing_time = np.mean(processing_times)
        std_processing_time = np.std(processing_times)
        max_processing_time = np.max(processing_times)
        min_processing_time = np.min(processing_times)
        
        # Simple recommendation logic based on problem size and characteristics
        recommendation = {}
        
        # For small problems, local search is often fast and effective
        if num_jobs <= 20 and num_machines <= 5:
            recommendation['algorithm'] = 'local_search'
            recommendation['params'] = {
                'neighborhood_size': min(50, num_jobs * num_machines),
                'step_size': 1
            }
            recommendation['reason'] = "Small problem size is well-suited for Local Search, which is fast and provides good solutions for simple problems."
        
        # For medium problems with moderate variability, simulated annealing often works well
        elif (num_jobs <= 50 and num_machines <= 20) or (std_processing_time / avg_processing_time > 0.5):
            recommendation['algorithm'] = 'simulated_annealing'
            recommendation['params'] = {
                'initial_temperature': max_processing_time * num_jobs * 0.5,
                'cooling_rate': 0.95,
                'num_iterations': min(5000, num_jobs * num_machines * 10),
                'cooling_type': 'continuous',
                'cooling_method': 'geometric',
                'step_size': 50,
                'stagnation_limit': 100,
                'reheating_factor': 0.1
            }
            recommendation['reason'] = "Medium-sized problem with significant variability in processing times. Simulated Annealing can escape local optima and explore the solution space effectively."
        
        # For problems with high variability, genetic algorithms are often good
        elif std_processing_time / avg_processing_time > 0.7:
            recommendation['algorithm'] = 'genetic'
            recommendation['params'] = {
                'population_size': min(200, num_jobs * 5),
                'iterations': min(500, num_jobs * num_machines * 5),
                'crossover_rate': 0.8,
                'mutation_rate': 0.2,
                'selection_type': 'tournament',
                'tournament_size': 3,
                'crossover_type': 'two_point',
                'mutation_type': 'inversion',
                'seed': 42
            }
            recommendation['reason'] = "Problem has high variability in processing times. Genetic Algorithm's population-based approach can effectively explore diverse solutions."
        
        # For large complex problems, ant system often works well
        else:
            recommendation['algorithm'] = 'ant_system'
            recommendation['params'] = {
                'alpha': 1.0,
                'beta': 2.0,
                'visibility_strat': 'total_makespan',
                'q': 2.0,
                'ro': 0.5,
                'm': min(50, num_jobs * 2),
                'sigma0': 0.1,
                'n': min(200, num_jobs * num_machines * 2),
                'e': 1.0
            }
            recommendation['reason'] = "Large complex problem. Ant System's collective intelligence approach can find good solutions by exploring promising regions of the search space."
        
        # Add problem analysis
        recommendation['problem_analysis'] = {
            'num_jobs': int(num_jobs),
            'num_machines': int(num_machines),
            'avg_processing_time': float(avg_processing_time),
            'std_processing_time': float(std_processing_time),
            'max_processing_time': int(max_processing_time),
            'min_processing_time': int(min_processing_time),
            'coefficient_of_variation': float(std_processing_time / avg_processing_time)
        }
        
        return jsonify(recommendation)
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)})

# Add this new route for algorithm performance analytics
@app.route('/algorithm_analytics', methods=['GET'])
def algorithm_analytics():
    if not results_history:
        return jsonify({'error': 'No optimization history available'})
    
    # Group results by algorithm
    algorithms = {}
    for result in results_history:
        algorithm = result['algorithm']
        if algorithm not in algorithms:
            algorithms[algorithm] = []
        algorithms[algorithm].append(result)
    
    # Calculate statistics for each algorithm
    analytics = {}
    for algorithm, results in algorithms.items():
        makespans = [float(r['makespan']) if isinstance(r['makespan'], np.number) else r['makespan'] for r in results]
        execution_times = [float(r['execution_time']) for r in results]
        
        analytics[algorithm] = {
            'count': len(results),
            'avg_makespan': sum(makespans) / len(makespans) if makespans else 0,
            'min_makespan': min(makespans) if makespans else 0,
            'max_makespan': max(makespans) if makespans else 0,
            'avg_execution_time': sum(execution_times) / len(execution_times) if execution_times else 0,
            'min_execution_time': min(execution_times) if execution_times else 0,
            'max_execution_time': max(execution_times) if execution_times else 0,
            'problem_instances': list(set(r['problem_instance'] for r in results))
        }
    
    # Generate performance comparison chart
    plt.figure(figsize=(10, 6))
    
    # Prepare data for box plots
    data = []
    labels = []
    for algorithm, stats in analytics.items():
        if algorithms[algorithm]:
            data.append([float(r['makespan']) if isinstance(r['makespan'], np.number) else r['makespan'] for r in algorithms[algorithm]])
            labels.append(algorithm)
    
    # Create box plot
    plt.boxplot(data, labels=labels)
    plt.title('Algorithm Performance Comparison')
    plt.ylabel('Makespan')
    plt.grid(axis='y', alpha=0.3)
    
    # Convert plot to base64 string
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    performance_chart = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    # Generate execution time comparison chart
    plt.figure(figsize=(10, 6))
    
    # Prepare data for box plots
    data = []
    for algorithm in labels:
        data.append([float(r['execution_time']) for r in algorithms[algorithm]])
    
    # Create box plot
    plt.boxplot(data, labels=labels)
    plt.title('Algorithm Execution Time Comparison')
    plt.ylabel('Execution Time (s)')
    plt.grid(axis='y', alpha=0.3)
    
    # Convert plot to base64 string
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    execution_time_chart = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return jsonify({
        'analytics': analytics,
        'performance_chart': performance_chart,
        'execution_time_chart': execution_time_chart
    })

@app.route('/analytics')
def analytics():
    return render_template('analytics.html')

if __name__ == '__main__':
    socketio.run(app, debug=True)