from EGM_Operator import  EGM_Operator
import numpy as np
import numba
from numba import jit
from numba import njit

@njit
def solve_EGM(m_matrix,model,    # Class with model information
                          m_in,c_in,v_in,ev_egm_in,        # Initial conditions
                          tol=1e-10,
                          max_iter=1000,
                          verbose=True,
                          print_skip=100):

    # Set up loop
    i = 0
    error = tol + 1

    while error > tol:
        
        
        output = EGM_Operator(m_in,c_in,v_in, ev_egm_in, model,m_matrix)        
        
        #solve for norm 'tween iterations 
        error = np.max(np.abs(v_in - output[2]))

        i += 1
        
        # print(f"Error at iteration {i} is {error}.")
        
        if verbose and i % print_skip == 0:
            print(f"Error at iteration {i} is {error}.")
            
        m_in = output[0]
        c_in = output[1]
        v_in = output[2]
        ev_egm_in = output[3]

 
    print(f"\nConverged in {i} iterations.")

    return [m_in,c_in,v_in,ev_egm_in]