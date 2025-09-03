import numpy as np
import numba
from numba import jit
from numba import njit
from solve_EGM import solve_EGM
from numba import njit, prange
from step_markov_chain import step_markov_chain

@njit(parallel=True)
def simulate_agents_parallel(labor_paths,seed_assets,prod_states,
                    m_egm_0,c_egm_0,v_egm_0,ev_egm_0,
                    seed_locations,asset_paths,location_paths, number_agents, number_steps,model_in,m_matrix):
    

    ## note here dcegm model in would be an argument containing most of things above


    """
    Simulate agents using expected value functions.
    """
    # Initialize parameters
    β = model_in.β
    r = model_in.r
    γ = model_in.γ
    ν = model_in.ν
    
    ## grids
    wage_grid = model_in.wage_grid
    s_grid = model_in.s_grid
    asset_grid = model_in.asset_grid
    locations = model_in.locations
    number_locations = model_in.number_locations 
    
    # reshape pi matrix
    pi_matrix = np.array([ [model_in.pi_array[0],model_in.pi_array[1] ],[ model_in.pi_array[2],model_in.pi_array[3] ] ]) 


    ## functions
    wealth_of_a = model_in.wealth_of_a
    u_c = model_in.u_c
    du_dc = model_in.du_dc
    du_dc_inv = model_in.du_dc_inv
    interp_exterp = model_in.interp_exterp

            
    # Fixed effects
    # fixed_effects = np.zeros(154)
    # fixed_effects[30] = theta[0]  # FE_China
    # fixed_effects[65] = theta[1]  # FE_India
    # Home_FE = theta[12]
   
    # Value function calculations (assumes dcegm_vxd is implemented)
    model_solutions = solve_EGM(m_matrix,model_in, m_egm_0,c_egm_0,v_egm_0,ev_egm_0)
    m_final = model_solutions[0]
    c_final = model_solutions[1]
    v_final = model_solutions[2]
    ev_final = model_solutions[3]

    ## Seed initial assets and locations

    # pre-allocate conditional returns
    v_xd = np.ones(number_locations)
  
   # Simulate agent paths
    for i in prange(number_steps):
        for j in prange(number_agents):
 
            
            ## state variables
            asset_value = asset_paths[i, j]
            location = location_paths[i, j]
            
             ##run markov chain step
            initial_state = labor_paths[i, j]
            s_state = step_markov_chain(pi_matrix, initial_state)
            labor_paths[i,j+1] = s_state

            wealth = (1 + r) * asset_value + s_grid[s_state] * wage_grid[location]
            
            for dest_count in prange(number_locations):
                ## track index of location-destination pairs
                index_vxd = location * number_locations + dest_count
                
                ##set up home dummy
                home_dummy = 1 if (location == dest_count) else 0
                Home_FE = 1 
                
                if wealth <= m_matrix[location, dest_count]:
                    v_xd[dest_count] = -np.inf
                elif wealth <= m_final[1, index_vxd , s_state]:
                    v_xd[dest_count] = (1 /  ν) * (np.log(wealth) + home_dummy*Home_FE +  β * ev_final[0, dest_count , s_state])
                else:
                    v_xd[dest_count] = (1 /  ν) * np.interp(wealth, m_final[1:, index_vxd , s_state], 
                                                                v_final[1:, index_vxd , s_state])
            
            max_return_vec = np.max(v_xd)
            recenter_returns = v_xd - max_return_vec
            rhs_2_mat_exp = np.exp(recenter_returns)
            CCP_matrix = rhs_2_mat_exp / np.sum(rhs_2_mat_exp)
            
            #simulate choice of next period location
            CCP_mat_CDF = np.cumsum(CCP_matrix)
            shock = np.random.uniform()
            location_choice = np.sum(CCP_mat_CDF < shock)     
            
            index_c_tomorrow = location * number_locations + location_choice
            c_xd = np.interp(wealth, m_final[:, index_c_tomorrow , s_state], c_final[:, index_c_tomorrow , s_state])
            
            ## OLD if wealth <= m_final[1, index_vxd , s_state]:
            if wealth <= m_final[1, index_c_tomorrow , s_state]:
                asset_paths[i + 1, j] = 0
            else:
                asset_paths[i + 1, j] = γ ** (-1) * (wealth - m_matrix[location , location_choice ] - c_xd)
            
            location_paths[i + 1, j] = location_choice
    
    K_vec = asset_paths[number_steps, :]
    L_vec = location_paths[number_steps, :]
    
    return K_vec, L_vec, asset_paths, location_paths
