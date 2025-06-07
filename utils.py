# utils.py
import numpy as np

EPSILON = 1e-9

class Node:
    def __init__(self, starting=None, finishing=None,
                 front_starting=None, front_finishing=None,
                 num_machines=0):
        self.starting = starting if starting is not None else [] # List of job indices
        self.finishing = finishing if finishing is not None else [] # List of job indices (in reverse order of processing)
        
        self.front_starting = front_starting if front_starting is not None else np.zeros(num_machines)
        self.front_finishing = front_finishing if front_finishing is not None else np.zeros(num_machines)
        
        self.bound = 0.0 # Lower bound (FBg_bound)
        self.guide_value = 0.0 # g_gap guide value

    def __lt__(self, other): # For sorting nodes by guide_value
        return self.guide_value < other.guide_value

    def __repr__(self):
        s_str = str(self.starting[:5]) + ('...' if len(self.starting) > 5 else '')
        f_str = str(self.finishing[:5]) + ('...' if len(self.finishing) > 5 else '')
        fs_str = np.array_str(self.front_starting[:5], precision=0, suppress_small=True) + ('...' if len(self.front_starting) > 5 else '')
        ff_str = np.array_str(self.front_finishing[:5], precision=0, suppress_small=True) + ('...' if len(self.front_finishing) > 5 else '')

        return (f"Node(S:{s_str}, F:{f_str}, "
                f"FS:{fs_str}, FF:{ff_str}, "
                f"B:{self.bound:.2f}, G:{self.guide_value:.2f})")

def calculate_makespan_of_complete_schedule(schedule, num_machines, proc_times_jm):
    num_scheduled_jobs = len(schedule)
    if num_scheduled_jobs == 0:
        return 0.0

    completion_times = np.zeros((num_scheduled_jobs, num_machines))

    job0_idx_in_problem = schedule[0]
    completion_times[0, 0] = proc_times_jm[job0_idx_in_problem, 0]
    for m_idx in range(1, num_machines):
        completion_times[0, m_idx] = completion_times[0, m_idx-1] + proc_times_jm[job0_idx_in_problem, m_idx]

    for j_seq_idx in range(1, num_scheduled_jobs):
        current_job_idx_in_problem = schedule[j_seq_idx]
        completion_times[j_seq_idx, 0] = completion_times[j_seq_idx-1, 0] + proc_times_jm[current_job_idx_in_problem, 0]
        for m_idx in range(1, num_machines):
            completion_times[j_seq_idx, m_idx] = max(completion_times[j_seq_idx, m_idx-1], 
                                                     completion_times[j_seq_idx-1, m_idx]) + \
                                                 proc_times_jm[current_job_idx_in_problem, m_idx]
    
    return float(completion_times[num_scheduled_jobs-1, num_machines-1])


def delta_makespan_swap(schedule, num_machines, proc_times_jm, completion_times, i, j):
    """
    Compute the makespan delta for swapping two jobs at positions i and j in the schedule.
    schedule: list of job indices
    num_machines: number of machines
    proc_times_jm: 2D array [job, machine] of processing times
    completion_times: current completion time matrix of shape (n_jobs, num_machines)
    i, j: positions in the schedule to swap (i < j)

    Returns:
        new_makespan: float, makespan after swap
        delta: float, change in makespan (new - old)
        new_completion_times: updated completion_times matrix for the swapped schedule
    """
    # Copy original completion times for rollback
    n = len(schedule)
    new_ct = completion_times.copy()

    # Swap jobs in a local copy of schedule
    s = schedule.copy()
    s[i], s[j] = s[j], s[i]

    # Recompute affected segment from position i to j
    # If i == 0, initialize first row; else reuse prefix
    start = i
    if start == 0:
        # first job's C on machine 0
        job_idx = s[0]
        new_ct[0, 0] = proc_times_jm[job_idx, 0]
        # first job across machines
        for m in range(1, num_machines):
            new_ct[0, m] = new_ct[0, m-1] + proc_times_jm[job_idx, m]
        start = 1

    # Recompute from start to j
    for idx in range(start, j+1):
        job_idx = s[idx]
        # machine 0 uses previous job on same machine
        new_ct[idx, 0] = new_ct[idx-1, 0] + proc_times_jm[job_idx, 0]
        # machines 1..M-1
        for m in range(1, num_machines):
            new_ct[idx, m] = max(new_ct[idx, m-1], new_ct[idx-1, m]) + proc_times_jm[job_idx, m]

    # For positions > j, can reuse old completion times
    for idx in range(j+1, n):
        new_ct[idx] = completion_times[idx]

    old_makespan = completion_times[n-1, num_machines-1]
    new_makespan = new_ct[n-1, num_machines-1]
    delta = new_makespan - old_makespan
    return new_makespan, delta, new_ct

def concatenate_schedule(starting_jobs, finishing_jobs_reversed):
    actual_finishing_jobs = list(reversed(finishing_jobs_reversed))
    return starting_jobs + actual_finishing_jobs

def calculate_bound_bi_directional(node, all_job_indices, num_machines, proc_times_jm):
    max_bound_val = 0.0
    scheduled_jobs_set = set(node.starting) | set(node.finishing)
    
    R_i_array = np.zeros(num_machines)
    
    has_unscheduled = False
    for job_idx in all_job_indices:
        if job_idx not in scheduled_jobs_set:
            has_unscheduled = True
            break
            
    if has_unscheduled:
        for m_idx in range(num_machines):
            sum_p_m = 0
            for job_k_idx in all_job_indices: 
                if job_k_idx not in scheduled_jobs_set: 
                     sum_p_m += proc_times_jm[job_k_idx, m_idx]
            R_i_array[m_idx] = sum_p_m
            
    for m_idx in range(num_machines):
        current_machine_bound = node.front_starting[m_idx] + R_i_array[m_idx] + node.front_finishing[m_idx]
        if current_machine_bound > max_bound_val:
            max_bound_val = current_machine_bound
            
    return float(max_bound_val)

def insert_forward(parent_node, job_to_insert_idx, num_machines, proc_times_jm):
    child_node = Node(num_machines=num_machines) # Creates new Node instance
    child_node.starting = parent_node.starting + [job_to_insert_idx]
    child_node.finishing = list(parent_node.finishing) 
    child_node.front_finishing = parent_node.front_finishing.copy() 

    new_front_starting = np.zeros(num_machines)
    new_front_starting[0] = parent_node.front_starting[0] + proc_times_jm[job_to_insert_idx, 0]
    for m_idx in range(1, num_machines):
        start_j_on_machine_m = max(new_front_starting[m_idx-1], parent_node.front_starting[m_idx])
        new_front_starting[m_idx] = start_j_on_machine_m + proc_times_jm[job_to_insert_idx, m_idx]
    child_node.front_starting = new_front_starting
    return child_node

def insert_backward(parent_node, job_to_insert_idx, num_machines, proc_times_jm):
    child_node = Node(num_machines=num_machines) # Creates new Node instance
    child_node.starting = list(parent_node.starting)
    child_node.finishing = parent_node.finishing + [job_to_insert_idx]
    child_node.front_starting = parent_node.front_starting.copy()

    new_front_finishing = np.zeros(num_machines)
    m_last_idx = num_machines - 1
    new_front_finishing[m_last_idx] = parent_node.front_finishing[m_last_idx] + proc_times_jm[job_to_insert_idx, m_last_idx]
    for m_idx in range(m_last_idx - 1, -1, -1):
        start_j_on_machine_m = max(new_front_finishing[m_idx+1], parent_node.front_finishing[m_idx])
        new_front_finishing[m_idx] = start_j_on_machine_m + proc_times_jm[job_to_insert_idx, m_idx]
    child_node.front_finishing = new_front_finishing
    return child_node

def calculate_g_gap(node, current_UB, node_LB, num_machines, proc_times_jm):
    num_start = len(node.starting)
    num_finish = len(node.finishing)

    if (num_start > 0 and num_finish == 0) or \
       (num_start == 0 and num_finish > 0):
        return float('inf') 
    
    if node_LB >= current_UB - EPSILON:
        return node_LB 

    gap = current_UB - node_LB
    if gap < EPSILON : 
        return node_LB

    term_bound_weight = current_UB / gap
    term_idle_weight = gap / current_UB

    sum_idle_proportions = 0.0
    
    if num_start > 0:
        for m_idx in range(num_machines):
            if node.front_starting[m_idx] > EPSILON:
                sum_proc_time_start_m = 0.0
                for job_k_idx in node.starting:
                    sum_proc_time_start_m += proc_times_jm[job_k_idx, m_idx]
                I_f_i = node.front_starting[m_idx] - sum_proc_time_start_m
                if I_f_i > EPSILON:
                    sum_idle_proportions += I_f_i / node.front_starting[m_idx]

    if num_finish > 0:
        for m_idx in range(num_machines):
            if node.front_finishing[m_idx] > EPSILON:
                sum_proc_time_finish_m = 0.0
                for job_k_idx in node.finishing:
                    sum_proc_time_finish_m += proc_times_jm[job_k_idx, m_idx]
                I_b_i = node.front_finishing[m_idx] - sum_proc_time_finish_m
                if I_b_i > EPSILON:
                    sum_idle_proportions += I_b_i / node.front_finishing[m_idx]
    
    g_gap_value = term_bound_weight * node_LB + term_idle_weight * sum_idle_proportions
    return float(g_gap_value)