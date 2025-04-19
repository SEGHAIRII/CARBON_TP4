import numpy as np
import random
from scipy.spatial.distance import euclidean
from Optimizer import AbstractOptimizer
import optuna
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
        self.visibility_strat = params.get('visibility_strat','total_makespan') # how is the visiblity calculated
        self.q = 50*(self.problem.num_machines+self.problem.num_jobs)*params.get('q',1 ) # phermone update intensity, by default it is a value that is relatively close to what an average makespan would look like
        self.ro = params.get('ro',0.3) # evaporation rate
        self.m = params.get('m',5) # number of ants
        self.sigma0 = params.get('sigma0',1.0) # initial pheromone value for all edges of the graph
        self.n = params.get('n',100) # number of iterations in total


        self.pheromoneGraph = np.full((self.problem.num_jobs, self.problem.num_jobs), self.sigma0)
        self.frames = []

    def local_makespan(self,j,path): # compute the total makespan for a job j
        return np.sum(self.problem.processing_times[:, j])
    def total_makespan(self,j,path): # compute the total makespan for all jobs in the path, ending with j
        result, _ = self.problem.evaluate(path+[j])
        return result
    

    def optimize(self):
        # before starting the construction process, we need a reference solution to compare our algorithm's result with, so let's generate a random permutation
        frames = [self.pheromoneGraph]
        current_solution = list(np.random.permutation(self.problem.num_jobs))
        current_makespan, _ = self.problem.evaluate(current_solution)
        functions = {
            "total_makespan": self.total_makespan,
            "local_makespan": self.local_makespan
        }
        visibility = functions[self.visibility_strat] # set the visiblity formula to use later
        print(self.n , '\n')
        print(self.q , '\n')
        for iteration in range(self.n):
            nb_jobs = self.problem.num_jobs
            deltaPheromon= np.zeros((self.problem.num_jobs, self.problem.num_jobs))
            average_makespan = 0
            ants_log = []
            for ant in range(self.m):
                available_jobs = list(range(nb_jobs))
                path = [] # start with an empty path
                first_step = np.random.randint(0, nb_jobs)
                path.append(first_step) # start at a random node in the graph
                available_jobs.remove(first_step) 
                while len(available_jobs) > 1: # there are nb_jobs-1 jobs left and the last job will be chosen anyways so we will run nb_jobs-2 iterations
                    
                    current_job = path[-1] # get the current job so far
                    # now we will calculate the probability distribution so that the ant can pick the next task according to that distrbution
                    score_list = [(self.pheromoneGraph[current_job,job]**self.alpha) * (1/visibility(job,path))**self.beta for job in available_jobs] 
                    total = sum(score_list)
                    score_list = np.array(score_list)
                    distribution = score_list / total
                    sampled_job_index = np.random.choice(len(distribution), p=distribution)
                    selected_job = available_jobs[sampled_job_index]
                    path.append(selected_job)
                    available_jobs.remove(selected_job)
                # now that the loop is done, we should end up with nb_jobs-1 jobs scheduled, leaving us with the last job that will be automatically added to the list
                path.append(available_jobs[0]) 
                path_makespan , _= self.problem.evaluate(path)
                if(path_makespan<current_makespan):
                    current_solution=path
                    current_makespan=path_makespan
                #print("ant path and makespan : ", path, " ", path_makespan)
                average_makespan = average_makespan + path_makespan
                # computing delta sigma for this particular ant for later updates
                deltaSigma=self.q/path_makespan
                for arc in range(len(path)-1):
                    deltaPheromon[path[arc],path[arc+1]]+=deltaSigma
                ants_log.append((path,path_makespan))
            best_iteration_path,best_iteration_path_makespan = max(ants_log, key=lambda x: x[1])

            # the elitism part
            for arc in range(len(path)-1):
                deltaPheromon[best_iteration_path[arc],best_iteration_path[arc+1]]+=self.e*self.q/best_iteration_path_makespan
            
            
            self.pheromoneGraph= self.pheromoneGraph * (1-self.ro) + deltaPheromon
            frames.append(self.pheromoneGraph)
            print("average makespan per batch : ", average_makespan/self.m)
        self.best_makespan = current_makespan
        self.best_solution = current_solution
        self.frames = frames




    def generate_video(self,matrices, output_file='heatmap_video.mp4', fps=5):
        n = matrices[0].shape[0]
        
        # Set up the figure
        fig, ax = plt.subplots()
        vmin = np.min(matrices)
        vmax = np.max(matrices)
        cax = ax.imshow(matrices[0], cmap='hot', interpolation='nearest', vmin=vmin, vmax=vmax)
        plt.axis('off')

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
        # return {
        #     'alpha': trial.suggest_float('alpha', 0.0, 5.0),
        #     'beta': trial.suggest_float('beta', 1.0, 5.0), 
        #     'visibility_strat': trial.suggest_categorical('visibility_strat',  ['total_makespan','local_makespan']), 
        #     'q': trial.suggest_float('q', 0.1, 100.0,log=True), #here we are setting the ratio to be multiplied by 50*(nb_jobs+nb_machines)
        #     'ro': trial.suggest_float('ro', 0.1, 0.9), 
        #     'm': trial.suggest_int('m', 5, 50), 
        #     'sigma0': trial.suggest_float("sigma0", 0.01, 1.0, log=True),
        #     'n': trial.suggest_categorical("n",[50,100,500])
        # }
        return {
            'alpha': trial.suggest_float('alpha', 1.0, 1.0),
            'beta': trial.suggest_float('beta', 1.0, 1.0), 
            'visibility_strat': trial.suggest_categorical('visibility_strat',  ['total_makespan']), 
            'q': trial.suggest_float('q', 1,1), #here we are setting the ratio to be multiplied by 50*(nb_jobs+nb_machines)
            'ro': trial.suggest_float('ro', 0.7,0.7), 
            'm': trial.suggest_int('m', 10, 10), 
            'sigma0': trial.suggest_float("sigma0", 1.0,1.0),
            'n': trial.suggest_categorical("n",[100])
        }
if __name__ == "__main__":

    # # Load the problem
    # problem = FlowShopProblem('./data/20_5_1.txt')

    # # Create an Optuna study to minimize makespan
    # study = optuna.create_study(direction='minimize')

    # # Define the optimization loop
    # def objective(trial):
    #     # Suggest parameters for the LocalSearchOptimizer
    #     params = AntSystemOptimizer.suggest_params(trial)
    #     optimizer = AntSystemOptimizer(problem, **params)
    #     optimizer.run()
    #     result = optimizer.get_results()
    #     print(result)
    #     return result['makespan']

    # # Optimize the objective function with Optuna
    # study.optimize(objective, n_trials=1)

    # # Print the best hyperparameters and result
    # print(f"Best Hyperparameters: {study.best_params}")
    # print(f"Best Makespan: {study.best_value}")

    # Load the problem
    problem = FlowShopProblem('./data/20_5_1.txt')
    params = {}
    optimizer = AntSystemOptimizer(problem, **params)
    optimizer.optimize()
    optimizer.generate_video(optimizer.frames,fps=5)


    # Print the best hyperparameters and result
    print(f"Best path: {optimizer.best_solution}")
    print(f"Best Makespan: {optimizer.best_makespan}")


        