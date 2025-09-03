import numpy as np
from numba import njit

@njit
def step_markov_chain(P, initial_state):
    """
    Simulate a Markov chain.

    Parameters:
    - P: Transition matrix (n_states x n_states), rows must sum to 1
    - initial_state: Integer, starting state index

    Returns:
    - next state
    """
    
    next_state =  np.empty( 1, dtype=np.int32) 
    current_state = initial_state
    transition_probs = P[current_state]
    next_state = np.searchsorted(np.cumsum(transition_probs), np.random.rand())

    return next_state
