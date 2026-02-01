import numpy as np

def generic_viterbi_algorithm(obs, start_p, trans_p, emit_p):
    """ Viterbi: Finds the most likely global path (contextual). """
    n_states = trans_p.shape[0]
    T = len(obs)
    
    v_table = np.zeros((n_states, T))
    backpointer = np.zeros((n_states, T), dtype=int)
    
    # Initialization
    for s in range(n_states):
        v_table[s, 0] = start_p[s] * emit_p[s, obs[0]]
        
    # Recursion
    for t in range(1, T):
        for s in range(n_states):
            trans_probs = [v_table[s_prev, t-1] * trans_p[s_prev, s] * emit_p[s, obs[t]] 
                           for s_prev in range(n_states)]
            v_table[s, t] = np.max(trans_probs)
            backpointer[s, t] = np.argmax(trans_probs)
            
    # Backtracking
    best_last_state = np.argmax(v_table[:, T-1])
    best_path = [best_last_state]
    
    for t in range(T-1, 0, -1):
        prev_state = backpointer[best_path[-1], t]
        best_path.append(prev_state)
        
    best_path.reverse()
    return best_path

def naive_greedy_algorithm(obs, emit_p):
    """ Greedy: Chooses the local maximum at each step. (Used mainly for Weather) """
    path = []
    for o in obs:
        best_state_now = np.argmax(emit_p[:, o])
        path.append(best_state_now)
    return path

def forward_backward_algorithm(obs, start_p, trans_p, emit_p):
    """
    Calculates the most likely state for each moment t,
    taking into account ALL observations (past and future).
    Returns the Marginally Maximized Path (MPM).
    """
    n_states = trans_p.shape[0]
    T = len(obs)

    # 1. FORWARD (Alpha)
    alpha = np.zeros((n_states, T))
    
    for s in range(n_states):
        alpha[s, 0] = start_p[s] * emit_p[s, obs[0]]
        
    for t in range(1, T):
        for s in range(n_states):
            sum_probs = 0
            for prev_s in range(n_states):
                sum_probs += alpha[prev_s, t-1] * trans_p[prev_s, s]
            
            alpha[s, t] = sum_probs * emit_p[s, obs[t]]
            
        c = np.sum(alpha[:, t])
        if c > 0: alpha[:, t] /= c

    # 2. BACKWARD (Beta)
    beta = np.zeros((n_states, T))
    beta[:, T-1] = 1.0
    
    for t in range(T-2, -1, -1):
        for s in range(n_states):
            sum_probs = 0
            for next_s in range(n_states):
                sum_probs += trans_p[s, next_s] * emit_p[next_s, obs[t+1]] * beta[next_s, t+1]
            beta[s, t] = sum_probs

    # 3. COMBINATION (Gamma / Posterior)
    posterior = np.zeros((n_states, T))
    path_mpm = []
    
    for t in range(T):
        raw_probs = alpha[:, t] * beta[:, t]
        norm_factor = np.sum(raw_probs)
        if norm_factor == 0: norm_factor = 1.0
            
        posterior[:, t] = raw_probs / norm_factor
        best_state = np.argmax(posterior[:, t])
        path_mpm.append(best_state)
        
    return path_mpm

def crf_inference_algorithm(obs, start_p, trans_p, emit_p):
    """ Simulates CRF inference (Max-Sum in log-space). """
    n_states = trans_p.shape[0]
    T = len(obs)
    
    eps = 1e-10
    w_start = np.log(start_p + eps)
    w_trans = np.log(trans_p + eps)
    w_emit  = np.log(emit_p + eps)
    
    score_table = np.zeros((n_states, T))
    backpointer = np.zeros((n_states, T), dtype=int)
    
    for s in range(n_states):
        score_table[s, 0] = w_start[s] + w_emit[s, obs[0]]
        
    for t in range(1, T):
        for s in range(n_states):
            candidate_scores = [score_table[s_prev, t-1] + w_trans[s_prev, s] + w_emit[s, obs[t]] 
                                for s_prev in range(n_states)]
            score_table[s, t] = np.max(candidate_scores)
            backpointer[s, t] = np.argmax(candidate_scores)
            
    best_last_state = np.argmax(score_table[:, T-1])
    best_path = [best_last_state]
    
    for t in range(T-1, 0, -1):
        prev_state = backpointer[best_path[-1], t]
        best_path.append(prev_state)
        
    best_path.reverse()
    return best_path