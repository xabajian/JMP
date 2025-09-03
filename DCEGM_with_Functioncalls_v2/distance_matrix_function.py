import numpy as np
import numba


def distance_matrix_function(theta_hat, distance_matrix, prod_grid):
    
    n_regions = len(prod_grid)

    # Fixed effects
    FE_China = theta_hat[0]
    FE_India = theta_hat[1]
    Home_FE = theta_hat[12]  # Not used in original code

    fixed_effects = np.zeros(154)
    fixed_effects[29] = FE_China  # MATLAB index 30
    fixed_effects[64] = FE_India  # MATLAB index 65

    origin_cost_parameters = theta_hat[2:6]
    dest_cost_parameters = theta_hat[6:10]
    distance_cost_param1 = theta_hat[10]
    distance_cost_param2 = theta_hat[11]

    # Origin country codes
    origin_codes = np.array([
        0, 2, 1, 1, 2, 1, 3, 3, 1, 0, 2, 3, 2, 0, 1, 1, 2, 2, 2, 3, 2, 0, 0, 0, 1,
        3, 0, 0, 2, 2, 2, 2, 3, 2, 3, 1, 1, 3, 1, 2, 2, 1, 1, 3, 0, 3, 0, 2, 3, 3,
        2, 0, 1, 3, 0, 3, 1, 0, 0, 1, 0, 1, 3, 3, 1, 1, 2, 1, 3, 3, 3, 3, 1, 2, 0,
        3, 0, 0, 2, 2, 1, 0, 2, 2, 2, 0, 0, 2, 0, 0, 2, 1, 1, 1, 0, 2, 0, 3, 3, 1,
        0, 1, 3, 3, 0, 2, 1, 1, 2, 1, 3, 3, 3, 3, 2, 0, 1, 3, 0, 2, 0, 3, 3, 1, 2,
        3, 3, 1, 0, 2, 1, 3, 3, 1, 0, 0, 2, 0, 2, 2, 1, 0, 1, 3, 3, 3, 2, 0, 1, 2,
        1, 0, 0, 0
    ])

    dest_codes = origin_codes.copy()

    cost_origin = np.array([origin_cost_parameters[i] for i in origin_codes])
    cost_dest = np.array([dest_cost_parameters[i] for i in dest_codes])

    # Build cost matrices
    origin_cost_mat = np.tile(cost_origin[:, np.newaxis], (1, n_regions))
    dest_cost_mat = np.tile(cost_dest[np.newaxis, :], (n_regions, 1))
    distance_cost_matrix = distance_cost_param1 * distance_matrix + distance_cost_param2 * np.square(distance_matrix)

    m_matrix = origin_cost_mat + dest_cost_mat + distance_cost_matrix
    np.fill_diagonal(m_matrix, 0)  # zero diagonal

    return m_matrix




