import numpy as np
import random
from scipy.spatial.distance import euclidean
from Optimizer import AbstractOptimizer
import optuna
from Problem import FlowShopProblem


class AntSystemOptimizer(AbstractOptimizer):
    def __init__(self,problem, **params):
        super().__init__(problem,**params)

        # the hyper parameters we will play with in Ant System
        self.problem = problem
        self.alpha = params.get('alpha',1.0) # pheromone influence
        self.beta = params.get('beta',2.0) # visibility influence
        self.visibility_strat = params('visibility_strat','total_makespan') # how is the visiblity calculated
        self.q = 50*(self.problem.num_machines+self.problem.num_jobs)*params('q',1 ) # phermone update intensity, by default it is a value that is relatively close to what an average makespan would look like
        self.ro = params('ro',0.5) # evaporation rate
        self.m = params('m',5) # number of ants
        self.sigma0 = params('sigma0',1.0) # initial pheromone value for all edges of the graph
        self.n = params('n',100) # number of iterations in total


        self.pheromoneGraph = np.full((self.problem.num_jobs, self.problem.num_jobs), self.sigma0)


    def local_makespan(self,j,path): # compute the total makespan for a job j
        return np.sum(self.problem.processing_times[:, j])
    def total_makespan(self,j,path): # compute the total makespan for all jobs in the path, ending with j
        result, _ = self.problem.evaluate(path+[j])
        return result
    

    def optimize(self):
        functions = {
            "total_makespan": self.total_makespan,
            "local_makespan": self.local_makespan
        }
        visibility = function[self.visibility_strat] # set the visiblity formula to use later
        for iteration in range(self.n):
            nb_jobs = self.problem.num_jobs
            for ant in range(self.m):
                available_jobs = list(range(nb_jobs))
                path = [] # start with an empty path
                first_step = np.random.randint(0, nb_jobs)
                path.append(first_step) # start at a random node in the graph
                available_jobs.remove(first_step) 
                for i in range(nb_jobs-2) : # there are nb_jobs-1 jobs left and the last job will be chosen anyways so we will run nb_jobs-2 iterations
                    current_job = path[-1] # get the current job so far
                    # now we will calculate the probability distribution so that the ant can pick the next task according to that distrbution
                    score_list = [(self.pheromoneGraph[current_job,job]**self.alpha) * (1/visibility(job,path))**self.beta for job in available_jobs] 
                    total = sum(score_list)
                    distribution = score_list/ total
                    sampled_job_index = np.random.choice(len(distribution), p=distribution)
                    selected_job = available_jobs[sampled_job_index]
                    path.append(selected_job)
                    available_jobs.remove(selected_job)
                path.append(available_jobs[0]) 


    
    @classmethod
    def suggest_params(cls, trial):
        #setting the different search space intervals
        return {
            'alpha': trial.suggest_float('alpha', 0.0, 5.0),
            'beta': trial.suggest_float('beta', 1.0, 5.0), 
            'visibility_strat': trial.suggest_categorical('visibility_strat',  ['total_makespan','local_makespan']), 
            'q': trial.suggest_float('q', 0.1, 100.0,log=True), #here we are setting the ratio to be multiplied by 50*(nb_jobs+nb_machines)
            'ro': trial.suggest_float('ro', 0.1, 0.9), 
            'm': trial.suggest_int('m', 5, 50), 
            'sigma0': trial.suggest_float("sigma0", 0.01, 1.0, log=True),
            'n': trial.suggest_categorical("n",[50,100,500,1000,2000])
        }
if __name__ == "__main__":

    # Load the problem
    problem = FlowShopProblem('./data/20_5_1.txt')

    # Create an Optuna study to minimize makespan
    study = optuna.create_study(direction='minimize')

    # Define the optimization loop
    def objective(trial):
        # Suggest parameters for the LocalSearchOptimizer
        params = AntSystemOptimizer.suggest_params(trial)
        optimizer = AntSystemOptimizer(problem, **params)
        optimizer.run()
        result = optimizer.get_results()
        print(result)
        return result['makespan']

    # Optimize the objective function with Optuna
    study.optimize(objective, n_trials=50)

    # Print the best hyperparameters and result
    print(f"Best Hyperparameters: {study.best_params}")
    print(f"Best Makespan: {study.best_value}")

        