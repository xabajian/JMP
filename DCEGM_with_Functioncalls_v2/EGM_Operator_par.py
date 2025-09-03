import numpy as np
import numba
from numba import jit
from numba import njit, prange


@njit(parallel=True)
def EGM_Operator_par(m_egm_in,c_egm_in,v_egm_in,ev_egm_in, model_in,m_matrix):

    """
    The Coleman-Reffett operator using EGM

    Based on John Stachurski's work https://python.quantecon.org/egm_policy_iter.html 
    along with Fyodor Iskhakov's extension for DDC models.
    """
    
    ## params
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

            
    # Allocate memory for new solutions array
    ev_egm_out = np.zeros((model_in.grid_size,model_in.number_locations,model_in.prod_states))
    c_egm_out = np.zeros_like(c_egm_in)
    m_egm_out = np.zeros_like(m_egm_in)
    v_egm_out = np.zeros_like(v_egm_in)
    v_xd = np.ones((2, number_locations))
    
    # Loop over labor endowment states (s = 0,1 corresponding to MATLAB's 1,2)
    #for s,labor in enumerate(s_grid):
    for s in prange(len(s_grid)):
        
        # Loop over current locations (origins)
        for j in prange(number_locations):
            
                # Loop over locations chosen for tomorrow (destinations)
                for k in prange(number_locations):
                    
                    # Compute index of origin
                    egm_index_today = j * number_locations  + k # (Python: 0-indexed)
                    # Compute index of destination
                    egm_index_tomorrow = k * number_locations  # (Python: 0-indexed)

                    
                    # Loop over asset grid points (MATLAB i = 1:length(a_grid))
                    for i in prange(len(asset_grid)):
                    
                        ## choose asset level tomorrow from exogenous grid
                        a_in = asset_grid[i]
                        omega_low = wealth_of_a(a_in,0,k) 
                        omega_high = wealth_of_a(a_in,1,k) 

                        # For each potential destination next period
                        for m in prange(number_locations):
                            
                            ##index my value functions
                            index_c_tomorrow = egm_index_tomorrow + m
                            
                    
                            ##interpolate values
                            
                            #np interp
                            #  np.interp
                            
                            ##my funciton
                            # interp_exterp
                            
                            v_xd[0, m] = (1 / ν) * interp_exterp(omega_low, m_egm_in[1:, index_c_tomorrow, 0], v_egm_in[1:, index_c_tomorrow, 0])
                            v_xd[1, m] = (1 / ν) * interp_exterp(omega_high, m_egm_in[1:, index_c_tomorrow, 1], v_egm_in[1:, index_c_tomorrow, 1])


                          
                            
                        ## Compute CCPs (conditional choice probabilities)
                        # with recentering
                        max_return_vec = np.max(v_xd)  # shape (2,1)
                        recenter_returns = v_xd - max_return_vec  # recenter for numerical stability
                        rhs_2_mat_exp = np.exp(recenter_returns)
                        
                        # no recentering
                        # rhs_2_mat_exp = np.exp(v_xd)

                        CCP_denominator_low = np.sum(rhs_2_mat_exp[0,:])
                        CCP_denominator_high = np.sum(rhs_2_mat_exp[1,:])
                        CCP_matrix_low = rhs_2_mat_exp[0,:] / CCP_denominator_low
                        CCC_matrix_high = rhs_2_mat_exp[1,:] / CCP_denominator_high

                        # Calculate c_prime_matrix: interpolated consumption for next period.
                        c_prime_matrix = np.zeros_like(v_xd)
                        
                        for h in prange(number_locations):
                            
                            index_c_tomorrow = egm_index_tomorrow + h

                            ## numpy bencmark
                            # np.interp
                            ## my version allowing for extrapolation
                            # interp_exterp
                            
                            
                            # Interpolate consumption for low state
                            c_prime_matrix[0, h] =  interp_exterp(omega_low, m_egm_in[:, index_c_tomorrow, 0],  c_egm_in[:, index_c_tomorrow, 0])
                                                   
                            # Interpolate consumption for high state
                            c_prime_matrix[1, h] =  interp_exterp(omega_high, m_egm_in[:, index_c_tomorrow, 1],  c_egm_in[:, index_c_tomorrow, 1])

                            
                        # Solve for consumption using the Euler equation.
                        u_prime_matrix = du_dc(c_prime_matrix)
                        # Sum over the consumption states (row-wise sum)
                        state_conditional_euler_low= np.sum(u_prime_matrix[0,:] * CCP_matrix_low)  # shape (2,)
                        state_conditional_euler_high = np.sum(u_prime_matrix[1,:] * CCC_matrix_high)  # shape (2,)
                        euler_RHS = pi_matrix[s, 0]* state_conditional_euler_low + pi_matrix[s, 1] * state_conditional_euler_high
                        
                        # Update consumption (note the shift by one row relative to the asset grid)
                        c_egm_out[i + 1, egm_index_today, s] = γ * du_dc_inv(euler_RHS*(1+r)* β)
                        # if  c_egm_out[i + 1, egm_index_today, s] < 0:
                        #     print("c is failing")
                            
                            
                        # Update required wealth (m_egm)
                        m_egm_out[i + 1, egm_index_today, s] = model_in.γ * a_in + c_egm_out[i + 1, egm_index_today, s]   +  m_matrix[j,k]
                        # if  m_egm_out[i + 1, egm_index_today, s] < 0:
                        #     print("m is failing")
                            
                            
                            
                        # Update expected value function (EV) with recentering 
                        # Here we take the log of the sum of exponentials and add the max (to reverse the earlier recentering)
                        EV_new = ν * (
                            pi_matrix[s, 0]*np.log(CCP_denominator_low)
                            + pi_matrix[s, 1]*np.log(CCP_denominator_high)
                            + max_return_vec)
                        
                        ev_egm_out[i,k,s]= EV_new
                        


                        # Add fixed effects / structural errors.
                        home_dummy = 1 if (j == k) else 0
                        Home_FE=1
                        
                        ## solve for updated v_egm 
                        if c_egm_out[i + 1, egm_index_today, s] > 0:
                            v_egm_out[i + 1, egm_index_today, s] = (
                                u_c(c_egm_out[i + 1, egm_index_today, s])
                                + home_dummy * Home_FE
                                +  β * EV_new
                            )
                        else:
                            v_egm_out[i + 1, egm_index_today, s] = -np.inf
                            

    #print(max_return_vec)
    return [m_egm_out,c_egm_out,v_egm_out,ev_egm_out]