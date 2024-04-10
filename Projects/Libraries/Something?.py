"""
Structure
    __init__:                                       Setup Structure
        settings:                                       Model Settings
        model_settings:                                 For customized models
        set_parameters:                                 Setup of model parameters
        create_transition_matrix:                        Form the Initial transition matrix
        
        For RSDC:
        univariate_density:                             Density of the univariate model 
        univariate_parameters:                          The parameters of the univariate model 
        calculate_standard_deviations:                  Calculate the standard deviations by univariate parameters
        calculate_standardised_residuals:               Calculates the residuals from standard deviations
        
    fit:                                            Runs E_Step, M_Step, Track
        E_Step:                                         Calculates Expectations
            get_densities:                                  Draw probability densities for the model. Shape [K,T,N], Using Densities
                Densities:                                      The Probability Density Function
            Forward_pass:                                   Calculate the Forward Probabilities
            Backward_pass:                                  Calculate the Backward Probabilities
            calculate_smoothed_probabilities:               Calculates Smoothed states & probabilities
            switch_states:                                  For multiple timeseries, ensures uniform prediction of states
            calculate_initial_state:                        Calculates the initial state probability
            calculate_transition_probabilities:             Calculates the Off Diagonal Transition Probabilities
 
        M_Step:                                         Maximizes Expectations
            Gamma_to_T:                                     Matrix with no diagonal values, For Scipy Minimize
            extract_off_diagonal_params:                    Get the off diagonal values of T (working) matrix
            estimate_model_parameters:                      Estimate the parameters of the model.
                estimate_closed_form:                           Calculates the closed form model parameters
                minimize_parameters:                            Alternatively use a minimization function
                    Objective_function:                             Sum the log likelihoods
                        get_minimization_densities:                     Draw log densities from parameters
                form_transition_matrix_from_paramseters:        Create T matrix from estimated parameters
                T_to_transition_matrix:                         Form the correct transition matrix from th T matrix
        
        Track:                                          To manage tracking estimation histories, breaking model etc.
"""



#     def calculate_transition_probabilities(self):
#         # Step 1: Aggregate transition probabilities
#         # Sum over all time steps to get total transition counts between states
#         total_transitions = self.v_hat.sum(axis=1)  # Sum across time dimension
        
#         # Normalize to get probabilities by dividing each transition count by the sum of transitions from the state
#         transition_probabilities = total_transitions / total_transitions.sum(axis=2, keepdims=True)

#         # Step 2 & 3: Since the T matrix you're working with is based on the off-diagonal elements,
#         # you'll need to convert these transition probabilities back to the T matrix form.
#         # Assuming T matrix construction similar to your `Gamma_to_T` function or
#         # using the provided utility functions if applicable.

#         # Convert transition_probabilities to T using an appropriate method
#         # For demonstration, assuming you want to convert directly and store it
#         self.transition_matrix = transition_probabilities

#         # Note: If T requires specific transformation from transition probabilities, you should apply it here.
#         # For instance, you might need to extract off-diagonal elements or apply logarithms,
#         # depending on how you define T in relation to transition probabilities.

#     def M_Step(self):
#         # Estimate parameters
#         self.estimate_model_parameters()

#         # Draw new densities
#         self.get_densities()

#         # Calculate the log likelihoods
#         self.calculate_log_likelihood()
        
#     def calculate_log_likelihood(self):
#         """
#         The log likelihood is the sum of 3 terms.
#         log_likelihood = term_1 + term_2 + term_3
#         term_1 = sum_{j=1}^n_states         u_hat[state, t=1] * np.log(initial_state_probability[j]) 

#         term_2 = sum_{j=1}^n_states
#                     sum_{j=1}^n_states 
#                         sum_{t=1}^T        v_hat[j,k,t] * log gamma[jk]
#         term_3 = sum_{j=1}^n_states 
#                         sum_{t=1}^T        u_hat[j,k,t] * log densitiy_j[t]
#         """
#         # Term 1
#         # self.u_hat[:, 0, :] has shape (K, N) representing the probability of being in each state at t=1 for all series
#         # np.log(self.initial_state_probabilities) has shape (N,) representing the log of the initial state probabilities
#         term_1 = np.sum(self.u_hat[:, 0, :] * np.log(self.initial_state_probabilities), axis=-1)

#         # Term 2
#         # np.log(self.transition_matrix) has shape (N, N)
#         # The sum is over the second to last time points, all transitions, and all series
#         log_transition_matrix = np.log(self.transition_matrix)[:, np.newaxis, :, :]  # Shape becomes (2, 1, 3, 3)
#         # Now, log_transition_matrix can be broadcast with self.v_hat for element-wise multiplication
#         term_2 = np.sum(self.v_hat * log_transition_matrix, axis=(-2, -1))
#         # term_2 = np.sum(self.v_hat * np.log(self.transition_matrix), axis=(-2, -1))

#         # Term 3
#         # self.densities has shape (K, T, N)
#         # We take the log of self.densities and then multiply by self.u_hat
#         # Term 3 for each series
#         term_3 = np.sum(self.u_hat * np.log(self.densities), axis=(-2, -1))

#         # Total Log-Likelihood
#         LLN_series = term_1 + np.sum(term_2, axis=-1) + term_3
#         print(LLN_series)
#         print(LLN_series.shape)
#         self.current_likelihood = LLN_series

#     def Gamma_to_T(self):
#         pass
#     def extract_off_diagonal_params(self):
#         pass
#     def estimate_model_parameters(self):
#         pass
#     def estimate_closed_form(self):
#         pass
#     def minimize_parameters(self):
#         pass
#     def Objective_function(self):
#         pass
#     def get_minimization_densities(self):
#         pass
#     def form_transition_matrix_from_paramseters(self):
#         pass
#     def T_to_transition_matrix(self):
#         pass

#     def Track(self, iteration):
#         # Log-Likelihood History
#         self.log_likelihood_history[iteration, :] = self.current_likelihood
#         # np.zeros((self.max_iterations, self.K))
#         self.initial_state_history[iteration] = self.initial_state_probabilities.copy()
#         self.probability_history[iteration] = self.transition_matrix.copy()

#     def finalize_parameters(self):
#         pass 
#     def summarize(self):
#         pass 
#     def plot_smoothed_states(self):
#         pass 
#     def plot_reults(self):
#         pass 
#     def plot_convergence(self):
#         pass 
#     def plot_residuals(self):
#         pass 
#     def plot_smoothed_history(self):
#         pass 






# #  ===========================================================
# #  |                AutoRegressive Model                     |
# #  ===========================================================




#     def track(self, iteration, ):
#         # Log-Likelihood History
#         self.log_likelihood_history[iteration, :] = self.current_likelihood
#         # np.zeros((self.max_iterations, self.K))
#         self.parameter_history[iteration] = self.model_parameters.copy()
#         # parameter_history(self.max_iterations, 3 (mu, phi, sigma), self.K, self.n_states)
#         self.initial_state_probabilities_history[iteration] = self.initial_state_probabilities.copy()
#         self.transition_matrix_history[iteration] = self.transition_matrix.copy()
        

#     def estimate_model_parameters(self):
#         # Iterate over each state to calculate parameters
#         for j in range(self.n_states):
#             # Weighted mean and variance for each series
#             for k in range(self.K):
#                 u_hat_jk = self.u_hat[k, :, j]  # Smoothed probabilities for state j in series k
                
#                 # Calculate the weighted mean for state j in series k
#                 weighted_sum = np.sum(u_hat_jk * self.data[k, :])
#                 total_weight = np.sum(u_hat_jk)
#                 self.model_parameters[0, k, j] = weighted_sum / total_weight  # Store mu
                
#                 # Calculate the weighted variance for state j in series k
#                 weighted_variance_sum = np.sum(u_hat_jk * (self.data[k, :] - self.model_parameters[0, k, j])**2)
#                 self.model_parameters[2, k, j] = np.sqrt(weighted_variance_sum / total_weight)  # Store sigma
                
#                 # Correcting AR(1) parameter estimation
#                 if total_weight > 1:  # Ensure there's enough data for estimation
#                     X = self.data[k, :-1]  # Observations at t-1
#                     Y = self.data[k, 1:]  # Observations at t
#                     u_hat_jk_shifted = u_hat_jk[1:]  # Shifted to align with X and Y
                    
#                     # Compute elements for AR(1) parameter estimation
#                     phi_numerator = np.sum(u_hat_jk_shifted * X * Y) - np.sum(u_hat_jk_shifted * X) * np.sum(u_hat_jk_shifted * Y) / np.sum(u_hat_jk_shifted)
#                     phi_denominator = np.sum(u_hat_jk_shifted * X**2) - (np.sum(u_hat_jk_shifted * X) ** 2) / np.sum(u_hat_jk_shifted)
                    
#                     self.model_parameters[1, k, j] = phi_numerator / phi_denominator if phi_denominator != 0 else 0  # Store phi

#     def finalize(self):
#         # Initialize a list to store optimization results for each series
#         optimization_results = []

#         # Optimize parameters for each series individually
#         for k in range(self.K):
#             # Extract initial parameters for the current series
#             initial_params = self.model_parameters[:, k, :].flatten()
#             bounds = [(-0.99, 0.99), (-0.99, 0.99), (0.01, None)] * self.n_states
            
#             # Perform optimization for the current series
#             result = minimize(self.objective_function, initial_params, args=(k,), bounds=bounds, method='L-BFGS-B')
            
#             if result.success:
#                 # Update model parameters for the current series with optimized values
#                 self.final_model_parameters[:, k, :] = result.x.reshape((3, self.n_states))
#                 # Append the successful result to the results list
#                 optimization_results.append((k, result))
#                 print('Success')
#             else:
#                 print(f"Optimization failed for series {k}:", result.message)
#                 # Optionally, you could also append the failed result to keep track of which series failed
#                 optimization_results.append((k, None))

#         # Store the optimization results in the instance for later use or analysis
#         self.optimization_results = optimization_results

#     def objective_function(self, params_flat, k):
#                 # Reshape flat parameter array back into parameter matrix for the current series
#         params = params_flat.reshape((3, self.n_states))
        
#         # Calculate densities based on current parameters for the current series
#         final_densities = self.get_final_densities(params, k)
        
#         # Calculate the negative log likelihood for the current series
#         nll = self.calculate_final_log_likelihood(final_densities, k)
        
#         return nll

#     def get_final_densities(self, parameters, k):
#         final_densities = np.zeros((self.T, self.n_states))
#         for t in range(self.T):
#             for n in range(self.n_states):
#                 final_densities[t, n] = self.final_density(parameters, k, t, n)
#         return final_densities

#     def final_density(self, parameters, k, t, n):
#         # Compute Gaussian PDF, using parameters for the current series
#         return (1. / (parameters[2, n] * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((self.data[k, t] - parameters[0, n] - parameters[1, n] * self.data[k, t-1]) / parameters[2, n]) ** 2) + 1e-6

#     def calculate_final_log_likelihood(self, final_densities, k):
#         # Calculate and return the negative log likelihood for the current series
#         # Implementation should use `final_densities` and focus on calculations for the current series `k`
#         # Adjust the calculations of term_1, term_2, and term_3 to work per series

#         """
#         The log likelihood is the sum of 3 terms.
#         log_likelihood = term_1 + term_2 + term_3
#         term_1 = sum_{j=1}^n_states         u_hat[state, t=1] * np.log(initial_state_probability[j]) 

#         term_2 = sum_{j=1}^n_states
#                     sum_{j=1}^n_states 
#                         sum_{t=1}^T        v_hat[j,k,t] * log gamma[jk]
#         term_3 = sum_{j=1}^n_states 
#                         sum_{t=1}^T        u_hat[j,k,t] * log densitiy_j[t]
#         """
#         # Term 1
#         # self.u_hat[:, 0, :] has shape (K, N) representing the probability of being in each state at t=1 for all series
#         # np.log(self.initial_state_probabilities) has shape (N,) representing the log of the initial state probabilities
#         term_1 = np.sum(self.u_hat[:, 0, :] * np.log(self.initial_state_probabilities), axis=-1)

#         # Term 2
#         # np.log(self.transition_matrix) has shape (N, N)
#         # The sum is over the second to last time points, all transitions, and all series
#         term_2 = np.sum(self.v_hat * np.log(self.transition_matrix), axis=(-2, -1))

#         # Term 3
#         # self.densities has shape (K, T, N)
#         # We take the log of self.densities and then multiply by self.u_hat
#         # Term 3 for each series
#         term_3 = np.sum(self.u_hat * np.log(final_densities), axis=(-2, -1))

#         # Total Log-Likelihood
#         LLN_series = term_1 + np.sum(term_2, axis=-1) + term_3
#         return - np.sum(LLN_series)






# #   ===========================================================
# #   |                         RSDC Model                      |
# #   ===========================================================
    

# class RSDCEM(Base):
#     def __init__(self, dataframe, univariate_parameters=None, squared=False, *args, **kwargs):
#         super().__init__(dataframe, *args, **kwargs)
#         # self.univariate_parameters = univariate_parameters
#         self.squared = squared

#         if univariate_parameters is None:
#             self.univariate_parameters = np.zeros((self.E, 3))  
#             self.estimate_univariate_parameters()
#         else:
#             self.univariate_parameters = univariate_parameters

#         self.correlation_parameters = int(self.E * (self.E - 1) / 2)
#         # Initialize parameters array
#         self.set_parameters()
#         self.calculate_standard_deviations()
#         self.calculate_standardized_residuals()
#         self.densities = np.zeros((self.K, self.T, self.n_states))
#         self.log_likelihood_history = np.zeros(self.max_iterations)

#     def set_parameters(self):
#         # Assuming self.correlation_parameters is already defined correctly
#         # Initialize rho as a numpy array with zeros for all states
#         self.rho = np.zeros(self.correlation_parameters * self.n_states)  # Adjusted for a one-dimensional array
        
#         # Set first half (corresponding to state 0) to 0.1
#         first_half_end = self.correlation_parameters  # Calculate the end index for the first half
#         self.rho[:first_half_end] = -0.3
        
#         # Set second half (corresponding to state 1) to 0.0
#         # This step might be redundant if the array is initialized with zeros, 
#         # but it's included for clarity and in case the initialization value changes.
#         second_half_start = first_half_end  # Calculate the start index for the second half
#         self.rho[second_half_start:] = 0.3
        
#         # Initialize parameter history
#         self.parameter_history = np.zeros((self.max_iterations, self.correlation_parameters, self.n_states))

#     def set_univariate_density(self):
#         # Sets the Density method to GARCH or ARMACH based on self.squared.
#         if self.squared:
#             self.univariate_Density = self.GARCH
#         else:
#             self.univariate_Density = self.ARMACH

#     def GARCH(self, params, data):
#         omega, alpha, beta = params
#         T = len(data)
#         sigma2 = np.zeros(T)
#         sigma2[0] = np.var(data)  # Initial variance

#         for t in range(1, T):
#             sigma2[t] = omega + alpha * data[t-1]**2 + beta * sigma2[t-1]
        
#         return np.abs(sigma2) + 1e-8

#     def ARMACH(self, params, data):
#         omega, alpha, beta = params
#         T = len(data)
#         sigma2 = np.zeros(T)
#         sigma2[0] = np.var(data)  # Initial variance

#         for t in range(1, T):
#             sigma2[t] = omega + alpha * np.abs(data[t-1]) + beta * sigma2[t-1]
        
#         return np.abs(sigma2) + 1e-8



#     def negative_log_likelihood(self, params, data, volatility_function):
#         sigma2 = volatility_function(params, data)
#         nll = np.sum(np.log(sigma2) + data**2 / sigma2)
#         return nll



#     def estimate_univariate_parameters(self):
#         # Loop over each time series to estimate its parameters
#         for k in range(self.E):
#             # Extract the time series data
#             data = self.data[k, :]
            
#             # Optimize parameters using the appropriate density function
#             if self.squared:  # GARCH model
#                 initial_guess = [0.1, 0.15, 0.6]  # Example initial guess for [omega, alpha, beta]
#                 result = minimize(self.negative_log_likelihood, initial_guess, args=(data, self.GARCH), method='L-BFGS-B', bounds=[(0.001, None), (0, 1), (0, 1)])
#             else:  # ARMACH model
#                 initial_guess = [0.1, 0.15, 0.6]  # Example initial guess for [omega, alpha, beta]
#                 result = minimize(self.negative_log_likelihood, initial_guess, args=(data, self.ARMACH), method='L-BFGS-B', bounds=[(0.001, None), (0, 1), (0, 1)])
            
#             # Store the optimized parameters
#             if result.success:
#                 self.univariate_parameters[k, :] = result.x
#             else:
#                 print(f"Parameter estimation failed for series {k}")


#     def calculate_standard_deviations(self):
#         # Preallocate sigma array with the shape of self.data
#         sigmas = np.zeros_like(self.data)

#         # Initial variance based on the historical data for each series
#         initial_variances = np.var(self.data, axis=1)

#         # Set initial variance for each series
#         for k in range(self.E):
#             sigmas[k, 0] = initial_variances[k]

#         # Calculate sigmas for each time t using the appropriate model
#         for t in range(1, self.T):
#             for k in range(self.E):
#                 if self.squared:
#                     # GARCH
#                     sigmas[k, t] = self.univariate_parameters[k, 0] + self.univariate_parameters[k, 1] * self.data[k, t-1]**2 + self.univariate_parameters[k, 2] * sigmas[k, t-1]
#                 else:
#                     # ARMACH
#                     sigmas[k, t] = self.univariate_parameters[k, 0] + self.univariate_parameters[k, 1] * np.abs(self.data[k, t-1]) + self.univariate_parameters[k, 2] * np.abs(sigmas[k, t-1])

#         # If squared=False, take the square root for GARCH standard deviations
#         if self.squared:
#             sigmas = np.sqrt(sigmas)

#         self.standard_deviations = sigmas


#     def calculate_standardized_residuals(self):
#         # The original method may have inaccuracies in inverting and multiplying matrices.
#         # Correct approach for element-wise division to get standardized residuals:
#         self.residuals = self.data / self.standard_deviations





#     def form_correlation_matrix_vectorized(self, initial_parameters):
#         # Takes Parameters, and forms a lost for each.
#         # Calculate the number of parameters for each state's correlation matrix
#         num_params_per_state = len(initial_parameters) // self.n_states
        
#         # Initialize arrays to store the results
#         R_matrix = np.zeros((self.n_states, self.E, self.E))
#         det_R = np.zeros(self.n_states)
#         inv_R = np.zeros((self.n_states, self.E, self.E))
#         # Assuming R_matrix is an array of shape (n_states, E, E)
#         # and self.standard_deviations is of shape (E, T),
#         # where E is the number of variables, and T is the number of time points.

#         for state in range(self.n_states):
#             R = R_matrix[state]  # Correlation matrix for the state
#             # Construct covariance matrix from R and standard deviations
#             std_dev_matrix = np.diag(self.standard_deviations[:, t])  # Assuming a specific time t
#             cov_matrix = std_dev_matrix @ R @ std_dev_matrix  # Covariance matrix
#             det_R[state] = np.linalg.det(cov_matrix)  # Determinant of the covariance matrix
#             inv_R[state] = np.linalg.inv(R_matrix[state])

#     def form_correlation_matrix(self, initial_parameters):
#         # Takes Parameters, and forms a lost for each.
#         # Calculate the number of parameters for each state's correlation matrix
#         num_params_per_state = len(initial_parameters) // self.n_states
        
#         # Initialize arrays to store the results
#         R_matrix = np.zeros((self.n_states, self.E, self.E))
#         det_R = np.zeros(self.n_states)
#         inv_R = np.zeros((self.n_states, self.E, self.E))
        
#         # Split the rhos into n_states equally long arrays
#         for state in range(self.n_states):
#             start_index = state * num_params_per_state
#             end_index = start_index + num_params_per_state
#             state_rhos = initial_parameters[start_index:end_index]
            
#             # Calculate the indices for the upper triangular part excluding the diagonal
#             triu_indices = np.triu_indices(self.E, 1)
            
#             # Fill the correlation matrix for this state
#             R_matrix[state][np.diag_indices(self.E)] = 1  # Diagonal elements are 1
#             R_matrix[state][triu_indices] = state_rhos  # Upper triangular off-diagonal elements
#             R_matrix[state][triu_indices[1], triu_indices[0]] = state_rhos  # Mirror to the lower triangular part
            
#             # Compute determinant and inverse for each state's correlation matrix
#             det_R[state] = np.linalg.det(R_matrix[state])
#             inv_R[state] = np.linalg.inv(R_matrix[state])
    
#         return R_matrix, det_R, inv_R


#     def get_densities(self):
#         # Get correlation matrix in each state
#         self.R_matrix, self.det_R, self.inv_R = self.form_correlation_matrix(self.rho)


#         for n in range(self.n_states):
#             for t in range(self.T):
#                 self.densities[:,t, n] = self.Density(t,n)
#                 # self.densities[:, t, n] = self.Density(t,n)


#     def Density(self, t, n):
#         term_1 = self.E * np.log(2 * np.pi)
#         D = np.diag(self.standard_deviations[:,t])
#         det_D = np.linalg.det(D)
#         det_D = np.max((det_D, 1e-9))
#         term_2 = 2 * np.log(det_D)  # Assuming standard_deviations is calculated
#         term_3 = np.log(self.det_R[n] + 1e-8)  # Ensure numerical stability
#         term_4 = self.residuals[:, t].T @ self.inv_R[n] @ self.residuals[:, t]

#         # Calculate the log likelihood for each k and t, stored as densities for visualization
#         return np.exp(-0.5 * (term_1 + term_2 + term_3 + term_4))


#     def estimate_model_parameters(self):
#         # Calculate the number of parameters needed to form the correlation matrix R
#         num_parameters = self.E * (self.E - 1) / 2

#         # Define initial_parameters
#         # For simplicity, starting with all parameters set to a small value close to 0,
#         # indicating initial low correlation
#         initial_parameters = self.rho
#         # constraints = {'type': 'ineq', 'fun': self.optimization_constraint}
#         bounds = [(-0.99, 0.99) for _ in range(int(num_parameters * 2))]
#         # self.calculate_standard_deviations()
#         # self.calculate_standardized_residuals()

#         #options={}
#         # During optimization, use a constraint
#         def objective_function(initial_parameters):
#             return self.objective_function_vectorized(initial_parameters)


#         self.result = minimize(
#             objective_function,  # Your objective function
#             initial_parameters,  # Initial guess of the parameters
#             method='TNC',
#             # constraints=constraints,
#             # options=options,
#             bounds=bounds
#         )
#         #print(self.result.x)
#         self.rho = self.result.x
#         self.model_parameters = self.result.x
#         self.R_matrix = self.form_correlation_matrix(self.rho)
#         self.current_likelihood = self.result.fun

#     def objective_function_vectorized(self, initial_parameters):
#         log_densities = self.get_param_densities(initial_parameters)

#         # Term 1
#         term_1 = np.sum(self.u_hat[:, 0] * np.log(self.initial_state_probabilities + 1e-8), axis=1)

#         # Term 2
#         log_transition_matrix = np.log(self.transition_matrix + 1e-8)
#         term_2 = np.sum(self.v_hat * log_transition_matrix[np.newaxis, np.newaxis, :, :], axis=(1, 2, 3))

#         # Term 3
#         term_3 = np.sum(self.u_hat * log_densities, axis=(1, 2))

#         # Combine terms to calculate the total Log-Likelihood for each series
#         LLN_series = term_1 + term_2 + term_3

#         return -LLN_series



#     def get_param_densities_vectorized(self, initial_parameters):
#         R_matrix, R_det, R_inv = self.form_correlation_matrix_vectorized(initial_parameters)

#         # Preparing for vectorized computation of log densities
#         # Assuming self.residuals is shaped (E, T), with E as the number of variables
#         term_1 = -0.5 * self.E * np.log(2 * np.pi)
        
#         # Vectorized computation of determinant and inverse of D (diagonal matrix of standard deviations)
#         log_det_D = np.log(self.standard_deviations**2).sum(axis=0)  # Assuming standard deviations are (E, T)
#         term_2 = -0.5 * log_det_D
        
#         # Term 3: Log determinant of R
#         term_3 = -0.5 * np.log(R_det + 1e-8)[:, np.newaxis]  # Shape (n_states, 1) for broadcasting
        
#         # Term 4: Quadratic term
#         # Reshaping residuals for broadcasting: New shape (E, 1, T)
#         residuals_reshaped = self.residuals[:, np.newaxis, :]
#         # R_inv is (n_states, E, E), need to perform batch matrix multiplication with residuals
#         # Transpose residuals to align with the operation: (T, E)
#         residuals_transposed = self.residuals.T

#         # Execute the quadratic form operation for each state and time point
#         # New einsum path: 'nij,tj,ti->nt'
#         # Note: 'ti->nt' might seem counterintuitive, but it correctly represents the operation intended
#         term_4 = -0.5 * np.einsum('nij,tj,ti->nt', R_inv, residuals_transposed, residuals_transposed)
        
#         # Summing terms to get log densities: Shape will be (n_states, T)
#         log_densities = term_1 + term_2 + term_3 + term_4
        
#         # Transpose to match expected shape: (1, T, n_states)
#         return log_densities.transpose((2, 1, 0))

