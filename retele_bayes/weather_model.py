import numpy as np

def get_params_weather_standard():
    """ 2 States: Hot, Cold """
    states = ["Hot", "Cold"]
    # Observations: number of ice creams eaten
    obs_vocab = ["1 ice cream", "2 ice creams", "3 ice creams"]
    
    # Initial probability vector (pi)
    start_p = np.array([0.8, 0.2])
    
    # Transition Matrix: P(State_t | State_t-1)
    trans_p = np.array([
        [0.7, 0.3], # Hot -> Hot, Hot -> Cold
        [0.4, 0.6]  # Cold -> Hot, Cold -> Cold
    ])
    
    # Emission Matrix: P(Observation | State)
    emit_p = np.array([
        [0.1, 0.4, 0.5], # Hot: Prob of eating 1, 2, or 3 ice creams
        [0.7, 0.2, 0.1]  # Cold: Prob of eating 1, 2, or 3 ice creams
    ])
    
    return states, obs_vocab, start_p, trans_p, emit_p

def get_params_weather_complex():
    """ 4 States: Very Hot -> Very Cold """
    states = ["Very Hot", "Hot", "Cold", "Very Cold"]
    obs_vocab = ["1 ice cream", "2 ice creams", "3 ice creams"]
    
    start_p = np.array([0.4, 0.4, 0.1, 0.1])
    
    trans_p = np.array([
        [0.65, 0.30, 0.05, 0.00],
        [0.20, 0.60, 0.20, 0.00],
        [0.00, 0.25, 0.50, 0.25],
        [0.00, 0.05, 0.30, 0.65]
    ])
    
    emit_p = np.array([
        [0.05, 0.15, 0.80], # Very Hot
        [0.10, 0.60, 0.30], # Hot
        [0.30, 0.60, 0.10], # Cold
        [0.80, 0.15, 0.05]  # Very Cold
    ])
    
    return states, obs_vocab, start_p, trans_p, emit_p