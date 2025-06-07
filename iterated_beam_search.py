import numpy as np
import random
from scipy.spatial.distance import euclidean
from Optimizer import AbstractOptimizer
import optuna
from Problem import FlowShopProblem
# main_ibs.py
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor


# Import necessary components from our other files
from utils import Node, calculate_makespan_of_complete_schedule, \
                  concatenate_schedule, calculate_bound_bi_directional, \
                  insert_forward, insert_backward, calculate_g_gap

import random
from collections import deque

# =========================== just utility functions=====================================================
def delta_makespan_swap(schedule, num_machines, proc_times_jm, completion_times, i, j):
    """
    Compute the makespan delta for swapping two jobs at positions i and j in the schedule.
    Returns new_makespan, delta, and the updated completion_times matrix.
    """
    n = len(schedule)
    # Copy original completion times for rollback
    new_ct = completion_times.copy()

    # Apply swap on a schedule copy
    s = schedule.copy()
    s[i], s[j] = s[j], s[i]

    # Determine start row for recomputation
    start = i
    # If swapping at position 0, recompute first job row
    if start == 0:
        job0 = s[0]
        new_ct[0, 0] = proc_times_jm[job0, 0]
        for m in range(1, num_machines):
            new_ct[0, m] = new_ct[0, m-1] + proc_times_jm[job0, m]
        start = 1

    # Recompute rows from start to j
    for idx in range(start, j+1):
        job_idx = s[idx]
        new_ct[idx, 0] = new_ct[idx-1, 0] + proc_times_jm[job_idx, 0]
        for m in range(1, num_machines):
            new_ct[idx, m] = max(new_ct[idx, m-1], new_ct[idx-1, m]) + proc_times_jm[job_idx, m]

    # Rows after j remain unchanged
    for idx in range(j+1, n):
        new_ct[idx] = completion_times[idx]

    old_mk = completion_times[n-1, num_machines-1]
    new_mk = new_ct[n-1, num_machines-1]
    delta = new_mk - old_mk
    return new_mk, delta, new_ct

def compute_completion_times(schedule, proc_times_jm, num_machines):
    n = len(schedule)
    completion_times = np.zeros((n, num_machines))
    completion_times[0, 0] = proc_times_jm[schedule[0], 0]
    for m in range(1, num_machines):
        completion_times[0, m] = completion_times[0, m - 1] + proc_times_jm[schedule[0], m]
    for i in range(1, n):
        job = schedule[i]
        completion_times[i, 0] = completion_times[i - 1, 0] + proc_times_jm[job, 0]
        for m in range(1, num_machines):
            completion_times[i, m] = max(completion_times[i, m - 1], completion_times[i - 1, m]) + proc_times_jm[job, m]
    return completion_times

def hill_climbing(initial_schedule, num_machines, proc_times_jm,  max_iterations=150):
    current_schedule = list(initial_schedule)
    n = len(current_schedule)
    completion_times = np.zeros((n, num_machines))
    # fill matrix
    first_job = current_schedule[0]
    completion_times[0,0] = proc_times_jm[first_job,0]
    for m in range(1, num_machines):
        completion_times[0,m] = completion_times[0,m-1] + proc_times_jm[first_job,m]
    for i in range(1, n):
        job = current_schedule[i]
        completion_times[i,0] = completion_times[i-1,0] + proc_times_jm[job,0]
        for m in range(1, num_machines):
            completion_times[i,m] = max(completion_times[i,m-1], completion_times[i-1,m]) + proc_times_jm[job,m]
    current_makespan = completion_times[-1, -1]
    
    for it in range(max_iterations):
        improved = False
        best_delta = 0
        best_swap = None
        best_ct = None
        best_makespan = current_makespan

        # Try all neighbor swaps
        for i in range(n - 1):
            for j in range(i + 1, n):
                new_mk, delta, new_ct = delta_makespan_swap(current_schedule, num_machines, proc_times_jm, completion_times, i, j)
                if new_mk < best_makespan:
                    improved = True
                    best_delta = delta
                    best_swap = (i, j)
                    best_ct = new_ct
                    best_makespan = new_mk

        if not improved:
            break  # No improvement found => local optimum

        # Apply the best move
        i, j = best_swap
        current_schedule[i], current_schedule[j] = current_schedule[j], current_schedule[i]
        completion_times = best_ct
        current_makespan = best_makespan

    return current_schedule, current_makespan


def create_root_node(num_machines, proc_times_jm, all_job_indices_list):
    # This function uses Node and calculate_bound_bi_directional from utils.py
    root = Node(num_machines=num_machines)
    root.bound = calculate_bound_bi_directional(root, all_job_indices_list, num_machines, proc_times_jm)
    return root

def generate_bi_directional_children(parent_node, current_UB,
                                     all_job_indices, num_jobs, num_machines, proc_times_jm,
                                     second_chance_prob=0.2):
    """
    Generate children by inserting unscheduled jobs forward/backward.
    If a child is pruned by bound, with probability p apply a random subsequence inversion
    on its partial schedule to give it a second chance.
    """
    f_children = []
    b_children = []

    scheduled_jobs_set = set(parent_node.starting) | set(parent_node.finishing)
    if len(scheduled_jobs_set) == num_jobs:
        return []

    unscheduled_job_indices = [j for j in all_job_indices if j not in scheduled_jobs_set]

    # Helper: attempt bound with second chance
    def try_bound_with_second_chance(child, current_UB):
        child.bound = calculate_bound_bi_directional(child, all_job_indices, num_machines, proc_times_jm)
        if child.bound < current_UB:
            return True  # passes normally
        # pruned: with probability p, invert a random subsequence and retry
        if random.random() < second_chance_prob:
            # choose subsequence in starting or finishing
            if child.starting and random.choice([True, False]):
                seq = child.starting
            else:
                seq = child.finishing
            if len(seq) >= 2:
                i, j = sorted(random.sample(range(len(seq)), 2))
                seq[i:j+1] = reversed(seq[i:j+1])
            # recompute bound
            child.bound = calculate_bound_bi_directional(child, all_job_indices, num_machines, proc_times_jm)
            return child.bound < current_UB
        return False

    # Generate forward insertions
    for job_idx in unscheduled_job_indices:
        child_f = insert_forward(parent_node, job_idx, num_machines, proc_times_jm)
        if try_bound_with_second_chance(child_f, current_UB + 1e-6):
            f_children.append(child_f)

    # Generate backward insertions
    for job_idx in unscheduled_job_indices:
        child_b = insert_backward(parent_node, job_idx, num_machines, proc_times_jm)
        if try_bound_with_second_chance(child_b, current_UB + 1e-6):
            b_children.append(child_b)

    # If one side empty, return the other
    if not f_children and not b_children:
        return []
    if not f_children:
        return b_children
    if not b_children:
        return f_children

    # Choose side with fewer children or better average bound
    sum_bounds_F = sum(c.bound for c in f_children)
    sum_bounds_B = sum(c.bound for c in b_children)
    if len(f_children) < len(b_children) or \
       (len(f_children) == len(b_children) and sum_bounds_F > sum_bounds_B):
        return f_children
    else:
        return b_children


def select_best_d_nodes(candidate_nodes, D_beam_width, current_UB, num_machines, proc_times_jm):
    # This function uses calculate_g_gap from utils.py and Node's __lt__ method
    if not candidate_nodes:
        return []

    for node_n in candidate_nodes:
        node_n.guide_value = calculate_g_gap(node_n, current_UB, node_n.bound, num_machines, proc_times_jm)
    
    candidate_nodes.sort() 
    return candidate_nodes[:D_beam_width]

def parallel_child_gen(args):
    parent_c_node, iteration_adaptive_UB, all_job_indices, num_jobs, num_machines, proc_times_jm = args
    return generate_bi_directional_children(parent_c_node, iteration_adaptive_UB,
                                            all_job_indices, num_jobs, num_machines, proc_times_jm)
    
    
class Iterated_Beam_Search(AbstractOptimizer):
    def __init__(self, problem,tracker = None ,**params):
        super().__init__(problem,  **params)
        self.initial_beam_width = int(params.get("initial_beam_width", 1))
        self.beam_width_factor = int(params.get("beam_width_factor", 2))
        self.max_ibs_iterations = int(params.get("max_ibs_iterations", 10))
        time_limit = params.get("time_limit_seconds", None)
        self.time_limit_seconds = int(time_limit) if time_limit is not '' else None
        self.tracker = tracker

    def optimize(self):
        num_jobs, num_machines, proc_times_jm = self.problem.num_jobs, self.problem.num_machines, self.problem.processing_times.T
        start_overall_time = time.time()

        overall_best_makespan = float('inf')
        overall_best_schedule = []

        all_job_indices = list(range(num_jobs))
        D = self.initial_beam_width

        print(f"Starting Iterative Beam Search. Max IBS Iterations: {self.max_ibs_iterations}, Time Limit: {self.time_limit_seconds}s\n")

        for ibs_iter_count in range(self.max_ibs_iterations):
            print(f'===========================iter = {ibs_iter_count}')
            iter_process_start_time = time.time()

            root_node = create_root_node(num_machines, proc_times_jm, all_job_indices)
            current_level_nodes = [root_node]

            if overall_best_makespan == float('inf'):
                current_UB_for_pruning = sum(proc_times_jm[i, j] for i in range(num_jobs) for j in range(num_machines))
            else:
                current_UB_for_pruning = overall_best_makespan * 1.1

            iter_best_schedule = []
            iter_best_makespan = float('inf')

            for level in range(num_jobs):
                if not current_level_nodes:
                    break

                # Prepare args for parallel execution
                task_args = [
                    (
                        parent_c_node,
                        current_UB_for_pruning * 1.2 if ibs_iter_count < 2 and D <= 5 else current_UB_for_pruning,
                        all_job_indices,
                        num_jobs,
                        num_machines,
                        proc_times_jm
                    )
                    for parent_c_node in current_level_nodes
                ]

                next_level_candidates_partial = []

                # Use multiprocessing to generate children in parallel
                with ProcessPoolExecutor() as executor:
                    results = executor.map(parallel_child_gen, task_args)

                    for children_nodes in results:
                        for child in children_nodes:
                            if len(child.starting) + len(child.finishing) == num_jobs:
                                schedule_solution = concatenate_schedule(child.starting, child.finishing)
                                makespan_solution = calculate_makespan_of_complete_schedule(schedule_solution, num_machines, proc_times_jm)

                                if makespan_solution < iter_best_makespan:
                                    iter_best_makespan = makespan_solution
                                    iter_best_schedule = schedule_solution
                            else:
                                next_level_candidates_partial.append(child)

                if not next_level_candidates_partial:
                    break

                current_level_nodes = select_best_d_nodes(next_level_candidates_partial, D, current_UB_for_pruning, num_machines, proc_times_jm)

            print(f'=====result before hill climbing = {iter_best_makespan} ======')
            if iter_best_makespan and len(iter_best_schedule) > 0:
                print("\n--- Starting Hill Climbing to further improve the solution ---")
                schedule_solution, makespan_solution = hill_climbing(iter_best_schedule, num_machines, proc_times_jm)
                if makespan_solution < overall_best_makespan:
                    overall_best_makespan = makespan_solution
                    overall_best_schedule = schedule_solution
                    current_UB_for_pruning = overall_best_makespan
            else:
                print("\n--- No valid schedule found for hill climbing in this iteration ---")

            current_total_execution_time = time.time() - start_overall_time
            iter_processing_duration = time.time() - iter_process_start_time

            print(f"--- IBS Iteration {ibs_iter_count + 1} (Beam Width D={D}) Completed ---")
            print(f"    Time for this iteration's processing: {iter_processing_duration:.4f} seconds")
            print(f"Best Makespan Found So Far: {overall_best_makespan if overall_best_makespan != float('inf') else 'N/A'}")
            printable_schedule = [j + 1 for j in overall_best_schedule] if overall_best_schedule else 'N/A'
            print(f"Best Schedule Found So Far (1-indexed jobs): {printable_schedule}")
            print(f"Total Execution Time Up To This Point: {current_total_execution_time:.4f} seconds")
            print("-" * 60)
            if self.tracker is not None:
                self.tracker.update(ibs_iter_count + 1, overall_best_makespan)

            if self.time_limit_seconds is not None and current_total_execution_time > self.time_limit_seconds:
                print("Time limit reached. Stopping IBS.")
                break

            D = D * self.beam_width_factor
            if D > (num_jobs ** 2) * 2 and num_jobs > 10 and D > 10000:
                print(f"Beam width D={D} is becoming very large. Stopping IBS to conserve memory.")
                break

        final_execution_time = time.time() - start_overall_time
        print("\n--- Final Result After All Iterations ---")
        print(f"Best Makespan: {overall_best_makespan if overall_best_makespan != float('inf') else 'N/A'}")
        printable_schedule_final = [j + 1 for j in overall_best_schedule] if overall_best_schedule else 'N/A'
        print(f"Best Schedule (1-indexed jobs): {printable_schedule_final}")
        print(f"Total Execution Time: {final_execution_time:.4f} seconds")

        self.best_solution = overall_best_schedule
        self.best_makespan = overall_best_makespan
        self.execution_time = final_execution_time
        if self.tracker is not None:
            self.tracker.complete(self.best_makespan, self.best_solution)
        
    @classmethod
    def suggest_params(cls, trial):
        pass
    
    
    
if __name__ == '__main__':
    problem = FlowShopProblem('./data/50_20_1.txt')
    optimizer = Iterated_Beam_Search(problem)
    optimizer.run()
    results = optimizer.get_results()
    print(f"Best makespan: {results['makespan']}")
    print(f"Best schedule: {results['schedule']}")
    print(f"Execution time: {results['execution_time']:.4f}s")
