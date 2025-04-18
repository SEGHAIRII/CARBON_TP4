import numpy as np
import time

class FlowShopProblem:
    def __init__(self, filepath):
        self.processing_times = self._read_file(filepath)
        self.num_machines, self.num_jobs = self.processing_times.shape

    def _read_file(self, filepath):
        with open(filepath, 'r') as file:
            lines = file.readlines()

        num_jobs, num_machines = map(int, lines[0].split())

        matrix = [list(map(int, line.strip().split())) for line in lines[1:]]

        matrix_np = np.array(matrix)
        assert matrix_np.shape == (num_machines, num_jobs), \
            f"Expected matrix of shape ({num_machines}, {num_jobs}), got {matrix_np.shape}"
        return matrix_np

    def evaluate(self, permutation):
        start_time = time.time()

        num_jobs = len(permutation)
        completion_times = np.zeros((num_jobs, self.num_machines))

        for i, job in enumerate(permutation):
            for machine in range(self.num_machines):
                processing_time = self.processing_times[machine][job]

                if i == 0 and machine == 0:
                    completion_times[i][machine] = processing_time
                elif i == 0:
                    completion_times[i][machine] = completion_times[i][machine - 1] + processing_time
                elif machine == 0:
                    completion_times[i][machine] = completion_times[i - 1][machine] + processing_time
                else:
                    completion_times[i][machine] = max(
                        completion_times[i - 1][machine],
                        completion_times[i][machine - 1]
                    ) + processing_time

        makespan = completion_times[-1][-1]
        end_time = time.time()
        execution_time = end_time - start_time

        return makespan, execution_time


    def get_num_jobs(self):
        return self.num_jobs

    def get_num_machines(self):
        return self.num_machines

    def get_processing_times(self):
        return self.processing_times.copy()
