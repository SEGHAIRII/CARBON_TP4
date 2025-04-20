import numpy as np
import random
from scipy.spatial.distance import euclidean
from Optimizer import AbstractOptimizer
import optuna
import time
from Problem import FlowShopProblem
import matplotlib.pyplot as plt
import cv2


class AntSystemOptimizer(AbstractOptimizer):
    def __init__(self,problem, **params):
        super().__init__(problem,**params)

        # the hyper parameters we will play with in Ant System
        self.problem = problem
        self.alpha = params.get('alpha',1.0) # pheromone influence
        self.beta = params.get('beta',2.0) # visibility influence
        self.visibility_strat = params.get('visibility_strat','local_makespan') # how is the visiblity calculated
        self.q = 50*(self.problem.num_machines+self.problem.num_jobs)*params.get('q',1.0 ) # phermone update intensity, by default it is a value that is relatively close to what an average makespan would look like
        self.ro = params.get('ro',0.5) # evaporation rate
        self.m = params.get('m',44) # number of ants
        self.sigma0 = params.get('sigma0',0.2) # initial pheromone value for all edges of the graph
        self.n = params.get('n',500) # number of iterations in total
        self.e = params.get('e',1.0) # elistist factor "pheromone boost to edges of the best path by iteration"

        # the pheromone graph will include a virtual task number -1 for all ants to start from
        self.pheromoneGraph = np.full((self.problem.num_jobs+1, self.problem.num_jobs+1), self.sigma0)
        self.frames = []

    def local_makespan(self,j,path): # compute the total makespan for a job j
        return np.sum(self.problem.processing_times[:, j])
    def total_makespan(self,j,path): # compute the total makespan for all jobs in the path, ending with j
        result= self.problem.evaluate(path+[j])
        return result
    

    def optimize(self):
        # before starting the construction process, we need a reference solution to compare our algorithm's result with, so let's generate a random permutation
        start_time = time.time()
        frames = [self.pheromoneGraph]
        current_solution = list(np.random.permutation(self.problem.num_jobs))
        current_makespan = self.problem.evaluate(current_solution)
        functions = {
            "total_makespan": self.total_makespan,
            "local_makespan": self.local_makespan
        }
        visibility = functions[self.visibility_strat] # set the visiblity formula to use later
        for iteration in range(self.n):
            nb_jobs = self.problem.num_jobs
            deltaPheromon= np.zeros((self.problem.num_jobs+1, self.problem.num_jobs+1))
            average_makespan = 0
            ants_log = []
            for ant in range(self.m):
                available_jobs = list(range(nb_jobs))
                path = [-1] # start always with the -1 task "the virtual initial task"
                while len(available_jobs) > 1: # we have nb_jobs available jobs, the ant will choose n-1 using decision rules, the last remaining job will be chosen anyways so we processed that later after the while loop
                    
                    current_job = path[-1] # get the current job so far
                    # now we will calculate the probability distribution so that the ant can pick the next task according to that distrbution
                    score_list = [(self.pheromoneGraph[current_job+1,job+1]**self.alpha) * ((1/visibility(job,path[1:]))**self.beta) for job in available_jobs] 
                    total = sum(score_list)
                    score_list = np.array(score_list)
                    # added .001 to avoid division by 0
                    distribution = (score_list+(0.001/len(available_jobs)) )/ (total+0.001)
                    sampled_job_index = np.random.choice(len(distribution), p=distribution)
                    selected_job = available_jobs[sampled_job_index]
                    path.append(selected_job)
                    available_jobs.remove(selected_job)
                # now that the loop is done, we should end up with nb_jobs-1 jobs scheduled, leaving us with the last job that will be automatically added to the list
                path.append(available_jobs[0]) 
                path_makespan = self.problem.evaluate(path[1:])
                if(path_makespan<current_makespan):
                    current_solution=path[1:]
                    current_makespan=path_makespan
                #print("ant path and makespan : ", path, " ", path_makespan)
                average_makespan = average_makespan + path_makespan
                # computing delta sigma for this particular ant for later updates
                deltaSigma=self.q/path_makespan
                for arc in range(len(path)-1):
                    deltaPheromon[path[arc]+1,path[arc+1]+1]+=deltaSigma
                ants_log.append((path,path_makespan))
            best_iteration_path,best_iteration_path_makespan = max(ants_log, key=lambda x: x[1])

            # the elitism part
            for arc in range(len(best_iteration_path)-1):
                deltaPheromon[best_iteration_path[arc]+1,best_iteration_path[arc+1]+1]+=self.e*self.q/best_iteration_path_makespan
            
            
            self.pheromoneGraph= self.pheromoneGraph * (1-self.ro) + deltaPheromon
            frames.append(self.pheromoneGraph)
            print("average makespan per batch : ", average_makespan/self.m, "   ", iteration)
        self.best_makespan = current_makespan
        self.best_solution = current_solution
        end_time = time.time()
        self.frames = frames
        self.execution_time = end_time-start_time





    # this visualises the evolution of pheromone intensity on the graph matrix
    def generate_video(self, matrices, output_file='heatmap_video.mp4', fps=5):
        n = matrices[0].shape[0]
        # Set up the figure
        fig, ax = plt.subplots()
        vmin = np.min(matrices)
        vmax = np.max(matrices)
        cax = ax.imshow(matrices[0], cmap='hot', interpolation='nearest', vmin=vmin, vmax=vmax)
        plt.axis('off')

        # Add colorbar to show intensity scale
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

        # Use Agg backend to render to a buffer
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        canvas = FigureCanvas(fig)

        # Grab frame dimensions
        canvas.draw()
        width, height = canvas.get_width_height()
        video = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        for matrix in matrices:
            cax.set_data(matrix)
            canvas.draw()
            img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape((height, width, 3))
            video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        video.release()
        plt.close()


    @classmethod
    def suggest_params(cls, trial):
        #setting the different search space intervals
        return {
            'alpha': trial.suggest_float('alpha', 0.0, 5.0),
            'beta': trial.suggest_float('beta', 1.0, 5.0), 
            'visibility_strat': trial.suggest_categorical('visibility_strat',  ['total_makespan','local_makespan']), 
            'q': trial.suggest_float('q', 0.1, 10.0,log=True), #here we are setting the ratio to be multiplied by 50*(nb_jobs+nb_machines)
            'ro': trial.suggest_float('ro', 0.1, 0.9), 
            'm': trial.suggest_int('m', 5, 50), 
            'sigma0': trial.suggest_float("sigma0", 0.01, 1.0, log=True),
            'n': trial.suggest_categorical("n",[50,100,500]),
            'e': trial.suggest_float("e",0.0,1.0)
        }

if __name__ == "__main__":


# there are two scenarios here, uncomment one of them only when testing

    #__1._________Run an entire HPO on a given instance__________________

    """

    # choose the instance
    problem = FlowShopProblem('./data/20_20_1.txt')

    # number of parameter samples to explore
    n_trials = 100

    # you can modify the intervals of searching for each parameter in the suggest_params class method

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
    study.optimize(objective, n_trials=n_trials)

    # Print the best hyperparameters and result
    print(f"Best Hyperparameters: {study.best_params}")
    print(f"Best Makespan: {study.best_value}")

    """




    #__2.__________Run a single instance with custom parameters__________________

    # choose the path to the instance
    path_to_instance = './data/20_5_1.txt'
    problem = FlowShopProblem(path_to_instance)

    # select the custom paramters for Ant system with elitism
    params = {
        'alpha': 1.0, # pheromon influence
        'beta': 2, # visiblity influence
        'q': 2.0, # pheromon intensity
        'ro': 0.5, # phermon evaporation factor
        'm': 10, # number of ants
        'sigma0': 0.2, # intial pheromon value on all edges
        'n':100, # number of iterations
        'visibility_strat': 'local_makespan', # visibilty strategy "either total_makespan or local_makespan"
        'e': 0.8 # elitism factor, 0 means original Ant system algorithm, 1 means max boost to best edges
    }

    optimizer = AntSystemOptimizer(problem, **params)
    optimizer.optimize()
    
    #uncomment this if you want to see a visualisation of the evolution of the graph matrix
    #optimizer.generate_video(optimizer.frames,fps=30)


    # This prints the path of the best solution, its makespan and the execution time in seconds
    print(f"Best path: {optimizer.best_solution}")
    print(f"Best Makespan: {optimizer.best_makespan}")
    print(f"Total Execution time: {optimizer.execution_time} seconds")


        