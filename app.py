import os
import io
import time
import json
import base64
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import threading
import traceback
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from flask_socketio import SocketIO, emit
import optuna
from optuna.visualization import plot_param_importances, plot_optimization_history

from Problem import FlowShopProblem
from AntSystem import AntSystemOptimizer
from Genetic import GeneticAlgorithmOptimizer
from LocalSearch_simple import LocalSearchOptimizer
from Simulated_annealing import SimulatedAnnealingOptimizer

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'carbon-flowshop-secret'
socketio = SocketIO(app)

# Store ongoing optimizations
ongoing_optimizations = {}

# Store optimization history
optimization_history = []

# Algorithm descriptions
algorithm_descriptions = {
    "ant_system": {
        "name": "Ant System",
        "description": "A population-based metaheuristic inspired by the foraging behavior of ants.",
        "strengths": [
            "Good at finding global optima in complex search spaces",
            "Adapts well to changing environments",
            "Parallelizable architecture"
        ],
        "weaknesses": [
            "Relatively slow convergence",
            "Parameter tuning can be difficult",
            "Memory and computationally intensive for large problems"
        ]
    },
    "genetic": {
        "name": "Genetic Algorithm",
        "description": "An evolutionary algorithm inspired by natural selection.",
        "strengths": [
            "Good at exploring large search spaces",
            "Can handle noisy environments",
            "Effective at finding approximate solutions quickly"
        ],
        "weaknesses": [
            "Can converge prematurely to local optima",
            "Performance depends on genetic operators and parameters",
            "No guarantee of finding the optimal solution"
        ]
    },
    "local_search": {
        "name": "Local Search",
        "description": "A simple hill-climbing algorithm that iteratively improves a solution.",
        "strengths": [
            "Simple to implement and understand",
            "Fast execution",
            "Low memory requirements"
        ],
        "weaknesses": [
            "Can get trapped in local optima",
            "Not effective for complex multimodal problems",
            "Sensitive to initial solution"
        ]
    },
    "simulated_annealing": {
        "name": "Simulated Annealing",
        "description": "A probabilistic technique inspired by the annealing process in metallurgy.",
        "strengths": [
            "Can escape local optima",
            "Suitable for large, complex search spaces",
            "Convergence can be controlled via cooling schedule"
        ],
        "weaknesses": [
            "Slower than basic local search",
            "Performance depends on cooling schedule",
            "Parameter tuning can be challenging"
        ]
    }
}

# Progress tracker class
class ProgressTracker:
    def __init__(self, optimizer_id, total_iterations):
        self.optimizer_id = optimizer_id
        self.total_iterations = total_iterations
        self.current_iteration = 0
        self.best_makespan = float('inf')
        self.best_solution = None
        self.status = 'running'
        self.message = ''
        self.start_time = time.time()
        
    def update(self, iteration, makespan):
        self.current_iteration = iteration
        if makespan < self.best_makespan:
            self.best_makespan = makespan
        
        progress = min(100, int((iteration / self.total_iterations) * 100)) if self.total_iterations > 0 else 0
        socketio.emit('optimization_progress', {
            'id': self.optimizer_id,
            'progress': progress,
            'message': f'Iteration {iteration} of {self.total_iterations} - Best makespan: {self.best_makespan:.2f}'
        })
    
    def complete(self, best_makespan, best_solution):
        self.status = 'complete'
        self.best_makespan = best_makespan
        self.best_solution = best_solution
        self.elapsed_time = time.time() - self.start_time
    
    def error(self, message):
        self.status = 'error'
        self.message = message
        socketio.emit('optimization_error', {
            'id': self.optimizer_id,
            'message': message
        })

# Routes
@app.route('/')
def index():
    # Get list of available problem instances
    problem_instances = sorted([f for f in os.listdir('data') if f.endswith('.txt')])
    
    # Define available algorithms
    algorithms = [
        {"id": "ant_system", "name": "Ant System"},
        {"id": "genetic", "name": "Genetic Algorithm"},
        {"id": "local_search", "name": "Local Search"},
        {"id": "simulated_annealing", "name": "Simulated Annealing"}
    ]
    
    return render_template('index.html', problem_instances=problem_instances, 
                          algorithms=algorithms, algorithm_descriptions=algorithm_descriptions)

@app.route('/history')
def history():
    return render_template('history.html', history=optimization_history)

@app.route('/compare')
def compare():
    return render_template('compare.html', history=optimization_history)

@app.route('/analytics')
def analytics():
    return render_template('analytics.html', history=optimization_history)

@app.route('/optuna')
def optuna_page():
    # Get list of available problem instances
    problem_instances = sorted([f for f in os.listdir('data') if f.endswith('.txt')])
    
    # Define available algorithms
    algorithms = [
        {"id": "ant_system", "name": "Ant System"},
        {"id": "genetic", "name": "Genetic Algorithm"},
        {"id": "local_search", "name": "Local Search"},
        {"id": "simulated_annealing", "name": "Simulated Annealing"}
    ]
    
    return render_template('optuna.html', problem_instances=problem_instances, 
                          algorithms=algorithms, algorithm_descriptions=algorithm_descriptions)

@app.route('/algorithm_info/<algorithm_id>')
def algorithm_info(algorithm_id):
    if algorithm_id in algorithm_descriptions:
        return jsonify(algorithm_descriptions[algorithm_id])
    else:
        return jsonify({"error": "Algorithm not found"})

@app.route('/get_algorithm_params', methods=['POST'])
def get_algorithm_params():
    data = request.json
    algorithm = data.get('algorithm')
    
    if algorithm == 'ant_system':
        params = [
            {
                "id": "alpha",
                "name": "Pheromone Influence (α)",
                "type": "number",
                "min": 0.1,
                "max": 5.0,
                "step": 0.1,
                "default": 1.0,
                "description": "Controls the influence of pheromone trails in the decision process"
            },
            {
                "id": "beta",
                "name": "Visibility Influence (β)",
                "type": "number",
                "min": 0.1,
                "max": 10.0,
                "step": 0.1,
                "default": 2.0,
                "description": "Controls the influence of heuristic information in the decision process"
            },
            {
                "id": "rho",
                "name": "Evaporation Rate (ρ)",
                "type": "number",
                "min": 0.01,
                "max": 0.99,
                "step": 0.01,
                "default": 0.5,
                "description": "Controls how quickly pheromone evaporates"
            },
            {
                "id": "q",
                "name": "Pheromone Intensity (Q)",
                "type": "number",
                "min": 1,
                "max": 1000,
                "step": 1,
                "default": 100,
                "description": "Controls the amount of pheromone deposited by ants"
            },
            {
                "id": "num_ants",
                "name": "Number of Ants",
                "type": "number",
                "min": 5,
                "max": 100,
                "step": 1,
                "default": 10,
                "description": "Number of ants to use in the colony"
            },
            {
                "id": "max_iterations",
                "name": "Max Iterations",
                "type": "number",
                "min": 10,
                "max": 1000,
                "step": 10,
                "default": 100,
                "description": "Maximum number of iterations to run"
            }
        ]
    elif algorithm == 'genetic':
        params = [
            {
                "id": "population_size",
                "name": "Population Size",
                "type": "number",
                "min": 10,
                "max": 500,
                "step": 10,
                "default": 100,
                "description": "Number of individuals in the population"
            },
            {
                "id": "crossover_rate",
                "name": "Crossover Rate",
                "type": "number",
                "min": 0.1,
                "max": 1.0,
                "step": 0.05,
                "default": 0.8,
                "description": "Probability of crossover between parents"
            },
            {
                "id": "mutation_rate",
                "name": "Mutation Rate",
                "type": "number",
                "min": 0.01,
                "max": 0.5,
                "step": 0.01,
                "default": 0.2,
                "description": "Probability of mutation in offspring"
            },
            {
                "id": "selection_method",
                "name": "Selection Method",
                "type": "select",
                "default": "tournament",
                "options": [
                    {
                        "value": "tournament",
                        "text": "Tournament Selection"
                    },
                    {
                        "value": "roulette",
                        "text": "Roulette Wheel Selection"
                    }
                ],
                "description": "Method used to select parents for reproduction"
            },
            {
                "id": "elitism",
                "name": "Elitism",
                "type": "number",
                "min": 0,
                "max": 20,
                "step": 1,
                "default": 2,
                "description": "Number of best individuals to carry over to next generation"
            },
            {
                "id": "max_generations",
                "name": "Max Generations",
                "type": "number",
                "min": 10,
                "max": 1000,
                "step": 10,
                "default": 100,
                "description": "Maximum number of generations to run"
            }
        ]
    elif algorithm == 'local_search':
        params = [
            {
                "id": "max_iterations",
                "name": "Max Iterations",
                "type": "number",
                "min": 10,
                "max": 10000,
                "step": 100,
                "default": 1000,
                "description": "Maximum number of iterations to run"
            },
            {
                "id": "neighborhood_size",
                "name": "Neighborhood Size",
                "type": "number",
                "min": 1,
                "max": 20,
                "step": 1,
                "default": 5,
                "description": "Number of neighbors to evaluate at each step"
            },
            {
                "id": "neighborhood_type",
                "name": "Neighborhood Type",
                "type": "select",
                "default": "swap",
                "options": [
                    {
                        "value": "swap",
                        "text": "Swap"
                    },
                    {
                        "value": "insert",
                        "text": "Insert"
                    }
                ],
                "description": "Type of neighborhood structure to use"
            }
        ]
    elif algorithm == 'simulated_annealing':
        params = [
            {
                "id": "initial_temperature",
                "name": "Initial Temperature",
                "type": "number",
                "min": 10,
                "max": 10000,
                "step": 10,
                "default": 1000,
                "description": "Starting temperature for annealing process"
            },
            {
                "id": "cooling_rate",
                "name": "Cooling Rate",
                "type": "number",
                "min": 0.5,
                "max": 0.999,
                "step": 0.001,
                "default": 0.95,
                "description": "Rate at which temperature decreases"
            },
            {
                "id": "max_iterations",
                "name": "Max Iterations",
                "type": "number",
                "min": 10,
                "max": 10000,
                "step": 100,
                "default": 1000,
                "description": "Maximum number of iterations to run"
            },
            {
                "id": "cooling_schedule",
                "name": "Cooling Schedule",
                "type": "select",
                "default": "exponential",
                "options": [
                    {
                        "value": "exponential",
                        "text": "Exponential"
                    },
                    {
                        "value": "linear",
                        "text": "Linear"
                    },
                    {
                        "value": "logarithmic",
                        "text": "Logarithmic"
                    }
                ],
                "description": "Function used to decrease temperature over time"
            }
        ]
    else:
        params = []
    
    return jsonify({"params": params})

@app.route('/get_problem_info/<filename>')
def get_problem_info(filename):
    try:
        # Load the problem
        problem_path = os.path.join('data', filename)
        problem = FlowShopProblem(problem_path)
        
        # Create processing times visualization
        plt.figure(figsize=(8, 6))
        sns.heatmap(problem.processing_times, cmap='viridis', annot=True, fmt='d')
        plt.title('Processing Times')
        plt.xlabel('Machines')
        plt.ylabel('Jobs')
        
        # Convert plot to base64 string
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        processing_times_viz = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return jsonify({
            "num_jobs": problem.num_jobs,
            "num_machines": problem.num_machines,
            "processing_times_viz": processing_times_viz
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)})

@app.route('/recommend_algorithm', methods=['POST'])
def recommend_algorithm():
    data = request.json
    problem_instance = data.get('problem_instance')
    
    if not problem_instance:
        return jsonify({"error": "Missing problem instance"})
    
    try:
        # Load the problem
        problem_path = os.path.join('data', problem_instance)
        problem = FlowShopProblem(problem_path)
        
        # Analyze problem characteristics
        num_jobs = problem.num_jobs
        num_machines = problem.num_machines
        total_processing_time = np.sum(problem.processing_times)
        avg_processing_time = total_processing_time / (num_jobs * num_machines)
        std_processing_time = np.std(problem.processing_times)
        processing_time_range = np.max(problem.processing_times) - np.min(problem.processing_times)
        
        # Problem analysis to inform recommendation
        problem_analysis = {
            "num_jobs": num_jobs,
            "num_machines": num_machines,
            "total_processing_time": int(total_processing_time),
            "avg_processing_time": float(avg_processing_time),
            "std_processing_time": float(std_processing_time),
            "processing_time_range": int(processing_time_range)
        }
        
        # Make recommendation based on problem characteristics
        # This is a simplified recommendation logic - in a real system, this would be more sophisticated
        if num_jobs <= 20 and num_machines <= 5:
            # Small problems - can use any algorithm, but local search is fast and effective
            algorithm = "local_search"
            reason = "For small problem instances, Local Search provides quick convergence to good solutions."
            params = {
                "max_iterations": 2000,
                "neighborhood_size": min(10, num_jobs // 2),
                "neighborhood_type": "insert"
            }
        elif num_jobs > 50 or num_machines > 10:
            # Large problems - genetic algorithms tend to perform well
            algorithm = "genetic"
            reason = "For large problem instances, Genetic Algorithms are effective at exploring complex search spaces."
            params = {
                "population_size": min(200, num_jobs * 5),
                "crossover_rate": 0.85,
                "mutation_rate": 0.2,
                "selection_method": "tournament",
                "elitism": 3,
                "max_generations": 100
            }
        elif std_processing_time > avg_processing_time * 0.5:
            # High variance in processing times - simulated annealing is good at escaping local optima
            algorithm = "simulated_annealing"
            reason = "With high variance in processing times, Simulated Annealing helps escape local optima."
            params = {
                "initial_temperature": 1000,
                "cooling_rate": 0.95,
                "max_iterations": 1000,
                "cooling_schedule": "exponential"
            }
        else:
            # Medium-sized, balanced problems - ant system works well
            algorithm = "ant_system"
            reason = "For medium-sized problems with balanced processing times, Ant System finds high-quality solutions."
            params = {
                "alpha": 1.0,
                "beta": 2.5,
                "rho": 0.5,
                "q": 100,
                "num_ants": num_jobs,
                "max_iterations": 100
            }
        
        return jsonify({
            "algorithm": algorithm,
            "reason": reason,
            "params": params,
            "problem_analysis": problem_analysis
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)})

@app.route('/run_optimization', methods=['POST'])
def run_optimization():
    data = request.json
    problem_instance = data.get('problem_instance')
    algorithm = data.get('algorithm')
    params = data.get('params', {})
    
    if not problem_instance or not algorithm:
        return jsonify({'error': 'Missing required parameters'})
    
    # Generate a unique ID for this optimization
    optimizer_id = f"{algorithm}_{problem_instance}_{int(time.time())}"
    
    # Load the problem
    problem_path = os.path.join('data', problem_instance)
    problem = FlowShopProblem(problem_path)
    
    # Convert params from string to appropriate types
    for key, value in params.items():
        try:
            if key in ['num_ants', 'max_iterations', 'max_generations', 'population_size', 'elitism', 'neighborhood_size']:
                params[key] = int(value)
            elif key in ['alpha', 'beta', 'rho', 'q', 'crossover_rate', 'mutation_rate', 'initial_temperature', 'cooling_rate']:
                params[key] = float(value)
        except (ValueError, TypeError):
            pass
    
    # Create a progress tracker
    iterations = params.get('max_iterations', 100)
    if algorithm == 'genetic':
        iterations = params.get('max_generations', 100)
    
    tracker = ProgressTracker(optimizer_id, iterations)
    ongoing_optimizations[optimizer_id] = tracker
    
    # Start the optimization in a separate thread
    def run_optimization_thread():
        try:
            # Create the appropriate optimizer
            if algorithm == 'ant_system':
                optimizer = AntSystemOptimizer(problem, **params)
            elif algorithm == 'genetic':
                optimizer = GeneticAlgorithmOptimizer(problem, **params)
            elif algorithm == 'local_search':
                optimizer = LocalSearchOptimizer(problem, **params)
            elif algorithm == 'simulated_annealing':
                optimizer = SimulatedAnnealingOptimizer(problem, **params)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            
            # Set up callback for progress updates
            def progress_callback(iteration, best_makespan, **kwargs):
                tracker.update(iteration, best_makespan)
                
            optimizer.set_callback(progress_callback)
            
            # Run the optimization
            optimizer.run()
            
            # Get results
            results = optimizer.get_results()
            
            # Create Gantt chart
            plt.figure(figsize=(12, 6))
            optimizer.plot_gantt_chart()
            
            # Convert chart to base64 string
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            gantt_chart = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            # Create solution visualization if applicable
            solution_viz = None
            if algorithm == 'ant_system':
                plt.figure(figsize=(10, 6))
                optimizer.plot_pheromone_matrix()
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                solution_viz = base64.b64encode(buf.read()).decode('utf-8')
                plt.close()
            
            # Mark optimization as complete
            tracker.complete(results['makespan'], optimizer.best_solution)
            
            # Emit results via socket
            socketio.emit('optimization_complete', {
                'id': optimizer_id,
                'best_makespan': results['makespan'],
                'elapsed_time': results['time'],
                'solution': [int(x) if isinstance(x, np.integer) else x for x in optimizer.best_solution]
            })
            
            # Emit visualization data
            socketio.emit('optimization_result', {
                'id': optimizer_id,
                'gantt_chart': gantt_chart,
                'solution_viz': solution_viz
            })
            
            # Add to history
            optimization_history.append({
                'id': optimizer_id,
                'algorithm': algorithm,
                'algorithm_name': algorithm_descriptions[algorithm]['name'],
                'problem_instance': problem_instance,
                'makespan': results['makespan'],
                'time': results['time'],
                'params': params,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'solution': [int(x) if isinstance(x, np.integer) else x for x in optimizer.best_solution]
            })
            
        except Exception as e:
            traceback.print_exc()
            tracker.error(str(e))
            
        finally:
            if optimizer_id in ongoing_optimizations:
                del ongoing_optimizations[optimizer_id]
    
    # Start the thread
    thread = threading.Thread(target=run_optimization_thread)
    thread.daemon = True
    thread.start()
    
    # Return the optimizer ID so the client can track progress
    return jsonify({'optimizer_id': optimizer_id})

@app.route('/optimization_status/<optimizer_id>')
def optimization_status(optimizer_id):
    if optimizer_id in ongoing_optimizations:
        tracker = ongoing_optimizations[optimizer_id]
        
        # Ensure we never divide by zero and provide safe defaults
        progress = 0
        if tracker.total_iterations > 0:
            progress = min(100, int((tracker.current_iteration / tracker.total_iterations) * 100))
        
        return jsonify({
            'status': tracker.status,
            'progress': progress,
            'current_iteration': tracker.current_iteration,
            'total_iterations': tracker.total_iterations,
            'best_makespan': tracker.best_makespan if tracker.best_makespan != float('inf') else None
        })
    else:
        # Check if it's in history
        for opt in optimization_history:
            if opt['id'] == optimizer_id:
                return jsonify({
                    'status': 'complete',
                    'progress': 100,
                    'best_makespan': opt['makespan']
                })
        
        return jsonify({'status': 'unknown'})

@app.route('/cancel_optimization/<optimizer_id>', methods=['POST'])
def cancel_optimization(optimizer_id):
    if optimizer_id in ongoing_optimizations:
        del ongoing_optimizations[optimizer_id]
        return jsonify({'success': True})
    else:
        return jsonify({'success': False, 'error': 'Optimization not found'})

@app.route('/export_history')
def export_history():
    format_type = request.args.get('format', 'csv')
    
    if format_type == 'csv':
        # Create a DataFrame from history
        data = []
        for opt in optimization_history:
            row = {
                'id': opt['id'],
                'algorithm': opt['algorithm_name'],
                'problem_instance': opt['problem_instance'],
                'makespan': opt['makespan'],
                'time': opt['time'],
                'timestamp': opt['timestamp']
            }
            # Add parameters
            for key, value in opt['params'].items():
                row[f"param_{key}"] = value
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Convert to CSV
        csv_data = df.to_csv(index=False)
        
        # Create a response
        buffer = io.StringIO()
        buffer.write(csv_data)
        buffer.seek(0)
        
        return send_file(
            io.BytesIO(buffer.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'flowshop_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
    
    return jsonify({'error': 'Unsupported format'})

@app.route('/run_optuna_optimization', methods=['POST'])
def run_optuna_optimization():
    data = request.json
    problem_instance = data.get('problem_instance')
    algorithm = data.get('algorithm')
    num_trials = int(data.get('num_trials', 20))
    
    if not problem_instance or not algorithm:
        return jsonify({'error': 'Missing required parameters'})
    
    # Generate a unique ID for this optimization
    optimizer_id = f"optuna_{algorithm}_{problem_instance}_{int(time.time())}"
    
    # Load the problem
    problem_path = os.path.join('data', problem_instance)
    problem = FlowShopProblem(problem_path)
    
    # Create a progress tracker
    tracker = ProgressTracker(optimizer_id, num_trials)
    ongoing_optimizations[optimizer_id] = tracker
    
    # Start the optimization in a separate thread
    def run_optuna_thread():
        try:
            # Create Optuna study
            study = optuna.create_study(direction='minimize')
            
            # Define objective function based on algorithm
            def objective(trial):
                if algorithm == 'ant_system':
                    params = {
                        'alpha': trial.suggest_float('alpha', 0.1, 5.0),
                        'beta': trial.suggest_float('beta', 0.1, 10.0),
                        'rho': trial.suggest_float('rho', 0.01, 0.99),
                        'q': trial.suggest_int('q', 1, 1000),
                        'num_ants': trial.suggest_int('num_ants', 5, min(100, problem.num_jobs * 2)),
                        'max_iterations': trial.suggest_int('max_iterations', 10, 100)
                    }
                    optimizer = AntSystemOptimizer(problem, **params)
                elif algorithm == 'genetic':
                    params = {
                        'population_size': trial.suggest_int('population_size', 10, 500),
                        'crossover_rate': trial.suggest_float('crossover_rate', 0.1, 1.0),
                        'mutation_rate': trial.suggest_float('mutation_rate', 0.01, 0.5),
                        'selection_method': trial.suggest_categorical('selection_method', ['tournament', 'roulette']),
                        'elitism': trial.suggest_int('elitism', 0, 20),
                        'max_generations': trial.suggest_int('max_generations', 10, 100)
                    }
                    optimizer = GeneticAlgorithmOptimizer(problem, **params)
                elif algorithm == 'local_search':
                    params = {
                        'max_iterations': trial.suggest_int('max_iterations', 10, 10000),
                        'neighborhood_size': trial.suggest_int('neighborhood_size', 1, 20),
                        'neighborhood_type': trial.suggest_categorical('neighborhood_type', ['swap', 'insert'])
                    }
                    optimizer = LocalSearchOptimizer(problem, **params)
                elif algorithm == 'simulated_annealing':
                    params = {
                        'initial_temperature': trial.suggest_int('initial_temperature', 10, 10000),
                        'cooling_rate': trial.suggest_float('cooling_rate', 0.5, 0.999),
                        'max_iterations': trial.suggest_int('max_iterations', 10, 10000),
                        'cooling_schedule': trial.suggest_categorical('cooling_schedule', ['exponential', 'linear', 'logarithmic'])
                    }
                    optimizer = SimulatedAnnealingOptimizer(problem, **params)
                
                optimizer.run()
                result = optimizer.get_results()
                
                # Update progress
                tracker.update(trial.number + 1, result['makespan'])
                
                return result['makespan']
            
            # Run optimization
            study.optimize(objective, n_trials=num_trials)
            
            # Get best parameters and results
            best_params = study.best_params
            best_value = study.best_value
            
            # Generate visualization of parameter importance
            param_importances = optuna.importance.get_param_importances(study)
            
            plt.figure(figsize=(10, 6))
            importance_values = list(param_importances.values())
            param_names = list(param_importances.keys())
            sorted_indices = np.argsort(importance_values)
            
            plt.barh([param_names[i] for i in sorted_indices], 
                     [importance_values[i] for i in sorted_indices])
            plt.xlabel('Importance')
            plt.title('Hyperparameter Importance')
            plt.tight_layout()
            
            # Convert plot to base64 string
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            param_importance_viz = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            # Generate optimization history plot
            plt.figure(figsize=(10, 6))
            trials = [t.value for t in study.trials]
            plt.plot(range(1, len(trials) + 1), trials)
            plt.axhline(y=best_value, color='r', linestyle='-', label=f'Best Value: {best_value:.2f}')
            plt.xlabel('Trial')
            plt.ylabel('Makespan')
            plt.title('Optimization History')
            plt.legend()
            plt.tight_layout()
            
            # Convert plot to base64 string
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            history_viz = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            # Run an optimization with the best parameters
            try:
                if algorithm == 'ant_system':
                    optimizer = AntSystemOptimizer(problem, **best_params)
                elif algorithm == 'genetic':
                    optimizer = GeneticAlgorithmOptimizer(problem, **best_params)
                elif algorithm == 'local_search':
                    optimizer = LocalSearchOptimizer(problem, **best_params)
                elif algorithm == 'simulated_annealing':
                    optimizer = SimulatedAnnealingOptimizer(problem, **best_params)
                
                optimizer.run()
                best_result = optimizer.get_results()
                best_solution = [int(x) if isinstance(x, np.integer) else x for x in optimizer.best_solution]
                
            except Exception as e:
                best_solution = []
                traceback.print_exc()
            
            # Mark optimization as complete
            tracker.complete(best_value, best_solution)
            
            # Emit final result via socket
            socketio.emit('optuna_complete', {
                'id': optimizer_id,
                'best_value': float(best_value),
                'best_params': best_params,
                'param_importance_viz': param_importance_viz,
                'history_viz': history_viz,
                'algorithm': algorithm,
                'algorithm_name': algorithm_descriptions[algorithm]['name'],
                'problem_instance': problem_instance
            })
            
            # Add to history
            optimization_history.append({
                'id': optimizer_id,
                'algorithm': f"optuna_{algorithm}",
                'algorithm_name': f"Optuna - {algorithm_descriptions[algorithm]['name']}",
                'problem_instance': problem_instance,
                'makespan': float(best_value),
                'time': time.time() - tracker.start_time,
                'params': best_params,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'solution': best_solution,
                'optuna_results': {
                    'num_trials': num_trials,
                    'param_importance': {k: float(v) for k, v in param_importances.items()},
                    'trials': [{'number': t.number, 'value': float(t.value), 'params': t.params} for t in study.trials]
                }
            })
            
        except Exception as e:
            traceback.print_exc()
            tracker.error(str(e))
            
        finally:
            if optimizer_id in ongoing_optimizations:
                del ongoing_optimizations[optimizer_id]
    
    # Start the thread
    thread = threading.Thread(target=run_optuna_thread)
    thread.daemon = True
    thread.start()
    
    # Return the optimizer ID so the client can track progress
    return jsonify({
        'optimizer_id': optimizer_id,
        'status': 'started'
    })

# Main
if __name__ == '__main__':
    socketio.run(app, debug=True)