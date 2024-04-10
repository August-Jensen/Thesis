import numpy as np
import pandas as pd

# Plotting
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

# Extra
from scipy.optimize import minimize
import time
from tqdm import tqdm
import warnings

# ===============================================================
# |         The Base Expectation Maximization Algorithm         |
# ===============================================================

'''
Function Overview:
    init:                                   Setup
    df_to_array:                            prepare data from a dataframe, and save labels
    create_transition_matrix:               Create a transition matrix (2 x 2) for now
    Density:                                Calculate densities, should take t, parameters, state
    get_denities:                           Create array of densities, dim = K, T, N        
    forward_pass:                           Calculate the forward probabilities, alpha
    backward_pass:                          Calculate the backward probabilities, beta
    calculate_smooothed_probabilities:      Use alpha & beta to calculate smoothed state probabilities & smoothed transition probabilities
    calculate_initial_states:               Calculate the initial state probabilities at time t=0
    estimate_model_parameters:              Estimate the model parameters for each time series. 
    calculate_log_likelihood:               Calculate the total log likelihood.                 
    E_step:                                 Runs forward & backward pass, and calculates smoothed probabilities
    M_step:                                 Estimates total log likeihood and model parameters
    track:                                  Manages parameter histories, and convergence.
    fit:                                    Runs E_step, M_step & track
    plot_smoothed_states                    Plotting smoothed states.
    plot_reults                             Plotting estimates
    plot_convergence                        Plotting convergence
    plot_residuals                          Plotting residuals
    plot_smoothed_history                   Create a 3D plot of a smoothed state, for time interval t=0 to t=250
    summarize                               gives estimated parameters, standard deviations etc.
    finalize_parameters                     Runs minimize with 'L-BFGS-B' for final parameter estimates, conditional on states, and for standard deviations, errors etc.

'''




class Base:
    # Function Structure:
    '''
    init:                                   Setup
        df_to_array:                            prepare data from a dataframe, and save labels
        create_transition_matrix:               Create a transition matrix (2 x 2) for now
        set_parameters:                         Creates the initial parameters
        Density:                                Calculate densities, should take t, parameters, state
    fit:                                    Runs E_step, M_step & track
        E_step:                                 Runs forward & backward pass, and calculates smoothed probabilities
            get_denities:                           Create array of densities, dim = K, T, N        
            forward_pass:                           Calculate the forward probabilities, alpha
            backward_pass:                          Calculate the backward probabilities, beta
            calculate_smooothed_probabilities:      Use alpha & beta to calculate smoothed state probabilities & smoothed transition probabilities
        M_step:                                 Estimates total log likeihood and model parameters
            calculate_initial_states:               Calculate the initial state probabilities at time t=0
            estimate_model_parameters:              Estimate the model parameters for each time series. 
            calculate_log_likelihood:               Calculate the total log likelihood.                 
        track:                                  Manages parameter histories, and convergence.
    
    finalize_parameters:                        Runs minimize with 'L-BFGS-B' on objective_function for final parameter estimates, conditional on states, and for standard deviations, errors etc.
        objective function:                         Turns flat array into matrix, then gets density array, and negative log likelihood.
        final_densities:                            Return Density based estimates 
        final_get_densities:                        Gets Density array for estimated densities
        final_log_likelihood:                       Use final_densities to calculate final log likelihood
    results:

        

        summarize:                              gives estimated parameters, standard deviations etc.
        plot_smoothed_states:                   Plotting smoothed states.
        plot_reults:                            Plotting estimates
        plot_convergence:                       Plotting convergence
        plot_residuals:                         Plotting residuals
        plot_smoothed_history:                  Create a 3D plot of a smoothed state, for time interval t=0 to t=250
        

    '''
    def __init__(self, dataframe, transition_guess=0.99, n_states=2, max_iterations=100, tolerance=1e-5):
        # Store Dataframe
        self.dataframe = dataframe

        # Model Settings
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        # Get data array, and series labels
        self.data, self.labels = self.df_to_array(self.dataframe)

        # Data Dimensions
        self.K, self.T = self.data.shape

        # Set number of states
        self.n_states = n_states

        # Transition Probabilities
        self.p_00, self.p_11 = transition_guess, transition_guess

        # Create Transition Matrix
        self.transition_matrix = self.create_transition_matrix(self.p_00, self.p_11)

        # Set initial state probability 1/n_states 
        self.initial_state_probabilities = np.ones(self.n_states) / self.n_states

        # # Model parameters
        # self.set_parameters()

        # Log-Likelihood History
        self.log_likelihood_history = np.zeros((self.max_iterations, self.K))
        self.initial_state_probabilities_history = np.zeros((self.max_iterations, self.n_states))
        self.transition_matrix_history = np.zeros((self.max_iterations, n_states, self.n_states))

        # Set Density array for model
        self.densities = np.zeros((self.K, self.T, self.n_states))


    def df_to_array(self, dataframe):
        # Create Numpy Array
        data_array = dataframe.to_numpy().T

        # Get Column Labels
        labels = dataframe.columns.tolist()

        return data_array, labels



    def set_parameters(self):
        pass




    def create_transition_matrix(self, p_00, p_11):
        transition_matrix = np.zeros([2,2])

        transition_matrix[0] = p_00, 1 - p_11
        transition_matrix[1] = 1 - p_00, p_11

        transition_matrix = transition_matrix
        # Return the Transition Matrix
        return transition_matrix




    def fit(self):
        for iteration in tqdm(range(self.max_iterations), desc='Fitting Model'):
            # Run the E-Step 
            self.E_step()

            # Run the M-Step 
            self.M_step()

            # Track parameters
            self.track(iteration)

            # Break if converged, using updated condition based on log likelihood change
            if iteration > 0 and np.abs(self.log_likelihood_history[iteration] - self.log_likelihood_history[iteration - 1]).max() < self.tolerance:
                print('The model converged!')
                break

    def E_step(self):
        # Get Densities
        self.get_densities()

        # Forward Pass
        self.forward_pass()

        # Backward Pass
        self.backward_pass()

        # Smoothing
        self.calculate_smoothed_probabilities()
        self.calculate_initial_states() 

        # Flip Smoothed States, if they are opposite of the first.
        # self.ensure_uniform_state_estimates()


    def get_densities(self):
        # Assuming self.densities is pre-initialized with the correct shape (K, T, n_states)
        K_indices, T_indices, state_indices = np.ogrid[:self.K, :self.T, :self.n_states]
        
        # Now use these indices to vectorize your Density calculation
        # This is a placeholder for how you might call a vectorized self.Density
        # You need to adapt self.Density to accept and handle vectorized inputs properly
        self.densities = self.Density(K_indices, T_indices, state_indices)        # for k in range(self.K):
        #     for state in range(self.n_states):
        #         for t in range(self.T):
        #             self.densities[k, t, state] = self.Density(k, t, state)




    def Density(self, k, t, n):
        # Compute Gausian PDF, using parameters
        # k = series, t = time, n = state
        # Ensure sigma is positive
        pass


    def forward_pass(self,):
        # Initialize the forward probability matrix forward_probabilities with shape (K, T, n_states)
        # and a scaling factor array with shape (K, T)
        forward_probabilities = np.zeros((self.K, self.T, self.n_states))
        scale_factors = np.zeros((self.K, self.T))
        
        # Initial step: Compute initial forward_probabilities values and scale them
        forward_probabilities[:, 0, :] = self.initial_state_probabilities * self.densities[:, 0, :]
        scale_factors[:, 0] = 1.0 / np.sum(forward_probabilities[:, 0, :], axis=1)
        forward_probabilities[:, 0, :] *= scale_factors[:, 0, np.newaxis]
        
        # Recursive step: Update forward_probabilities values for t > 0
        for t in range(1, self.T):
            # Vectorized update of forward_probabilities using matrix multiplication for transition probabilities
            for k in range(self.K):
                forward_probabilities[k, t, :] = np.dot(forward_probabilities[k, t-1, :], self.transition_matrix) * self.densities[k, t, :]
            
            # Scale forward_probabilities values to prevent underflow
            scale_factors[:, t] = 1.0 / np.sum(forward_probabilities[:, t, :], axis=1)
            forward_probabilities[:, t, :] *= scale_factors[:, t, np.newaxis]
        
        self.forward_probabilities = forward_probabilities
        self.scale_factors = scale_factors


    def backward_pass(self,):
        # Initialize the backward probability matrix backward_probabilities with shape (K, T, n_states)
        backward_probabilities = np.zeros((self.K, self.T, self.n_states))
        
        # Initial step: Set the last backward_probabilities values to 1 and scale them by the last scale factors
        backward_probabilities[:, self.T-1, :] = 1.0 * self.scale_factors[:, self.T-1, np.newaxis]
        
        # Recursive step: Update backward_probabilities values from T-2 down to 0
        for t in range(self.T-2, -1, -1):
            for k in range(self.K):
                # Vectorized update of backward_probabilities using matrix multiplication for transition probabilities
                # Note: Need to multiply by the densities for time t+1
                backward_probabilities[k, t, :] = np.dot(self.transition_matrix, (self.densities[k, t+1, :] * backward_probabilities[k, t+1, :]))
            
            # Scale backward_probabilities values to prevent underflow, using the same scale factors as forward pass
            backward_probabilities[:, t, :] *= self.scale_factors[:, t, np.newaxis]
        
        self.backward_probabilities = backward_probabilities




    def calculate_smoothed_probabilities(self):
        # Smoothed state probabilities
        numerator = self.forward_probabilities * self.backward_probabilities  # Element-wise multiplication
        denominator = numerator.sum(axis=2, keepdims=True)  # Sum over states for normalization
        self.u_hat = numerator / denominator
        
        # Precompute for smoothed transitions to avoid recomputation
        alpha_shifted = np.roll(self.forward_probabilities, shift=1, axis=1)  # Shift forward_probabilities by 1 to align t-1 with t
        alpha_shifted[:, 0, :] = 0  # Set initial shifted values to 0 as there's no t-1 for the first element
        
        # Smoothed transition probabilities
        numerator_v = alpha_shifted[:, :-1, :, np.newaxis] * self.transition_matrix[np.newaxis, np.newaxis, :, :] * self.densities[:, 1:, np.newaxis, :] * self.backward_probabilities[:, 1:, np.newaxis, :]
        denominator_v = numerator_v.sum(axis=(2, 3), keepdims=True) + 1e-6 # Sum over both j and k states for normalization
        self.v_hat = numerator_v / denominator_v


    def calculate_initial_states(self):
        # Calculate the mean of the smoothed state probabilities at t=0 across all K series
        initial_state_probabilities = self.u_hat[:, 0, :].mean(axis=0)
        self.initial_state_probabilities = initial_state_probabilities

    def ensure_uniform_state_estimates(self):
        pass

    def M_step(self):
        # Estimate parameters
        self.estimate_model_parameters()

        # Draw new densities
        self.get_densities()

        # Calculate the log likelihoods
        self.calculate_log_likelihood()
        




    def estimate_model_parameters(self):
        # # Iterate over each state to calculate parameters
        # for j in range(self.n_states):
        #     # Weighted mean and variance for each series
        #     for k in range(self.K):
        #         u_hat_jk = self.u_hat[k, :, j]  # Smoothed probabilities for state j in series k
                
        #         # Calculate the weighted mean for state j in series k
        #         weighted_sum = np.sum(u_hat_jk * self.data[k, :])
        #         total_weight = np.sum(u_hat_jk)
        #         self.model_parameters[0, k, j] = weighted_sum / total_weight  # Store mu
                
        #         # Calculate the weighted variance for state j in series k
        #         weighted_variance_sum = np.sum(u_hat_jk * (self.data[k, :] - self.model_parameters[0, k, j])**2)
        #         self.model_parameters[2, k, j] = np.sqrt(weighted_variance_sum / total_weight)  # Store sigma
                
        #         # Correcting AR(1) parameter estimation
        #         if total_weight > 1:  # Ensure there's enough data for estimation
        #             X = self.data[k, :-1]  # Observations at t-1
        #             Y = self.data[k, 1:]  # Observations at t
        #             u_hat_jk_shifted = u_hat_jk[1:]  # Shifted to align with X and Y
                    
        #             # Compute elements for AR(1) parameter estimation
        #             phi_numerator = np.sum(u_hat_jk_shifted * X * Y) - np.sum(u_hat_jk_shifted * X) * np.sum(u_hat_jk_shifted * Y) / np.sum(u_hat_jk_shifted)
        #             phi_denominator = np.sum(u_hat_jk_shifted * X**2) - (np.sum(u_hat_jk_shifted * X) ** 2) / np.sum(u_hat_jk_shifted)
                    
        #             self.model_parameters[1, k, j] = phi_numerator / phi_denominator if phi_denominator != 0 else 0  # Store phi
        pass


    def calculate_log_likelihood(self):
        """
        The log likelihood is the sum of 3 terms.
        log_likelihood = term_1 + term_2 + term_3
        term_1 = sum_{j=1}^n_states         u_hat[state, t=1] * np.log(initial_state_probability[j]) 

        term_2 = sum_{j=1}^n_states
                    sum_{j=1}^n_states 
                        sum_{t=1}^T        v_hat[j,k,t] * log gamma[jk]
        term_3 = sum_{j=1}^n_states 
                        sum_{t=1}^T        u_hat[j,k,t] * log densitiy_j[t]
        """
        # Term 1
        # self.u_hat[:, 0, :] has shape (K, N) representing the probability of being in each state at t=1 for all series
        # np.log(self.initial_state_probabilities) has shape (N,) representing the log of the initial state probabilities
        term_1 = np.sum(self.u_hat[:, 0, :] * np.log(self.initial_state_probabilities), axis=-1)

        # Term 2
        # np.log(self.transition_matrix) has shape (N, N)
        # The sum is over the second to last time points, all transitions, and all series
        term_2 = np.sum(self.v_hat * np.log(self.transition_matrix), axis=(-2, -1))

        # Term 3
        # self.densities has shape (K, T, N)
        # We take the log of self.densities and then multiply by self.u_hat
        # Term 3 for each series
        term_3 = np.sum(self.u_hat * np.log(self.densities), axis=(-2, -1))

        # Total Log-Likelihood
        LLN_series = term_1 + np.sum(term_2, axis=-1) + term_3
        self.current_likelihood = LLN_series



    def track(self, iteration, ):
        # Log-Likelihood History
        self.log_likelihood_history[iteration, :] = self.current_likelihood
        # np.zeros((self.max_iterations, self.K))
        self.initial_state_probabilities_history[iteration] = self.initial_state_probabilities.copy()
        self.transition_matrix_history[iteration] = self.transition_matrix.copy()



    def finalize_parameters(self):
        pass 




    def summarize(self):
        pass 




    def plot_smoothed_states(self):
        pass 




    def plot_reults(self):
        pass 




    def plot_convergence(self):
        pass 




    def plot_residuals(self):
        pass 




    def plot_smoothed_history(self):
        pass 
























# ===============================================================
# |               Algorithm for the AR model                    |
# ===============================================================

class AREM(Base):
    """docstring for AREM"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.set_parameters()

    def set_parameters(self):
        # Setup Model Parameters
        self.mu = np.zeros((self.K, self.n_states))
        self.phi = np.zeros((self.K, self.n_states))
        self.sigma = np.zeros((self.K, self.n_states))  

        # Initialize a 4D array: 3 (for mu, phi, sigma) x K x n_states
        self.model_parameters = np.zeros((3, self.K, self.n_states))
        self.final_model_parameters = np.zeros((3, self.K, self.n_states))
        # Set history for each iteration + 1, all 3 parameters
        self.parameter_history = np.zeros((self.max_iterations,3, self.K , self.n_states))
        for k in range(self.K):
            mean_k = np.mean(self.data[k, :])
            std_k = np.sqrt(np.var(self.data[k, :]))

            for state in range(self.n_states):
                # Perturb mean and standard deviation for diversity across states
                self.model_parameters[0, k, state] = mean_k + (state - self.n_states / 2) * 0.01  # mu
                self.model_parameters[2, k, state] = std_k + (state - self.n_states / 2) * 0.1  # sigma

        # for k in range(self.K):
        #     mean_k = np.mean(self.data[k, :])
        #     std_k = np.sqrt(np.var(self.data[k, :]))
            
        #     for state in range(self.n_states):
        #         # Perturb mean and standard deviation for diversity across states
        #         self.mu[k, state] = mean_k + (state - self.n_states / 2) * 0.01
        #         self.sigma[k, state] = std_k + (state - self.n_states / 2) * 0.1




    def Density(self, k, t, n):
        # Compute Gausian PDF, using parameters
        # k = series, t = time, n = state
        # Ensure sigma is positive
        return (1. / (self.model_parameters[2,k, n] * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((self.data[k,t] - self.model_parameters[0,k, n] - self.model_parameters[1,k, n] * self.data[k, t-1]) / self.model_parameters[2,k, n]) ** 2) + 1e-6
        

    def track(self, iteration, ):
        # Log-Likelihood History
        self.log_likelihood_history[iteration, :] = self.current_likelihood
        # np.zeros((self.max_iterations, self.K))
        self.parameter_history[iteration] = self.model_parameters.copy()
        # parameter_history(self.max_iterations, 3 (mu, phi, sigma), self.K, self.n_states)
        self.initial_state_probabilities_history[iteration] = self.initial_state_probabilities.copy()
        self.transition_matrix_history[iteration] = self.transition_matrix.copy()
        pass

    def estimate_model_parameters(self):
        # Iterate over each state to calculate parameters
        for j in range(self.n_states):
            # Weighted mean and variance for each series
            for k in range(self.K):
                u_hat_jk = self.u_hat[k, :, j]  # Smoothed probabilities for state j in series k
                
                # Calculate the weighted mean for state j in series k
                weighted_sum = np.sum(u_hat_jk * self.data[k, :])
                total_weight = np.sum(u_hat_jk)
                self.model_parameters[0, k, j] = weighted_sum / total_weight  # Store mu
                
                # Calculate the weighted variance for state j in series k
                weighted_variance_sum = np.sum(u_hat_jk * (self.data[k, :] - self.model_parameters[0, k, j])**2)
                self.model_parameters[2, k, j] = np.sqrt(weighted_variance_sum / total_weight)  # Store sigma
                
                # Correcting AR(1) parameter estimation
                if total_weight > 1:  # Ensure there's enough data for estimation
                    X = self.data[k, :-1]  # Observations at t-1
                    Y = self.data[k, 1:]  # Observations at t
                    u_hat_jk_shifted = u_hat_jk[1:]  # Shifted to align with X and Y
                    
                    # Compute elements for AR(1) parameter estimation
                    phi_numerator = np.sum(u_hat_jk_shifted * X * Y) - np.sum(u_hat_jk_shifted * X) * np.sum(u_hat_jk_shifted * Y) / np.sum(u_hat_jk_shifted)
                    phi_denominator = np.sum(u_hat_jk_shifted * X**2) - (np.sum(u_hat_jk_shifted * X) ** 2) / np.sum(u_hat_jk_shifted)
                    
                    self.model_parameters[1, k, j] = phi_numerator / phi_denominator if phi_denominator != 0 else 0  # Store phi

    def finalize(self):
        # Initialize a list to store optimization results for each series
        optimization_results = []

        # Optimize parameters for each series individually
        for k in range(self.K):
            # Extract initial parameters for the current series
            initial_params = self.model_parameters[:, k, :].flatten()
            bounds = [(-0.99, 0.99), (-0.99, 0.99), (0.01, None)] * self.n_states
            
            # Perform optimization for the current series
            result = minimize(self.objective_function, initial_params, args=(k,), bounds=bounds, method='L-BFGS-B')
            
            if result.success:
                # Update model parameters for the current series with optimized values
                self.final_model_parameters[:, k, :] = result.x.reshape((3, self.n_states))
                # Append the successful result to the results list
                optimization_results.append((k, result))
                print('Success')
            else:
                print(f"Optimization failed for series {k}:", result.message)
                # Optionally, you could also append the failed result to keep track of which series failed
                optimization_results.append((k, None))

        # Store the optimization results in the instance for later use or analysis
        self.optimization_results = optimization_results

    def objective_function(self, params_flat, k):
                # Reshape flat parameter array back into parameter matrix for the current series
        params = params_flat.reshape((3, self.n_states))
        
        # Calculate densities based on current parameters for the current series
        final_densities = self.get_final_densities(params, k)
        
        # Calculate the negative log likelihood for the current series
        nll = self.calculate_final_log_likelihood(final_densities, k)
        
        return nll

    def get_final_densities(self, parameters, k):
        final_densities = np.zeros((self.T, self.n_states))
        for t in range(self.T):
            for n in range(self.n_states):
                final_densities[t, n] = self.final_density(parameters, k, t, n)
        return final_densities

    def final_density(self, parameters, k, t, n):
        # Compute Gaussian PDF, using parameters for the current series
        return (1. / (parameters[2, n] * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((self.data[k, t] - parameters[0, n] - parameters[1, n] * self.data[k, t-1]) / parameters[2, n]) ** 2) + 1e-6

    def calculate_final_log_likelihood(self, final_densities, k):
        # Calculate and return the negative log likelihood for the current series
        # Implementation should use `final_densities` and focus on calculations for the current series `k`
        # Adjust the calculations of term_1, term_2, and term_3 to work per series

        """
        The log likelihood is the sum of 3 terms.
        log_likelihood = term_1 + term_2 + term_3
        term_1 = sum_{j=1}^n_states         u_hat[state, t=1] * np.log(initial_state_probability[j]) 

        term_2 = sum_{j=1}^n_states
                    sum_{j=1}^n_states 
                        sum_{t=1}^T        v_hat[j,k,t] * log gamma[jk]
        term_3 = sum_{j=1}^n_states 
                        sum_{t=1}^T        u_hat[j,k,t] * log densitiy_j[t]
        """
        # Term 1
        # self.u_hat[:, 0, :] has shape (K, N) representing the probability of being in each state at t=1 for all series
        # np.log(self.initial_state_probabilities) has shape (N,) representing the log of the initial state probabilities
        term_1 = np.sum(self.u_hat[:, 0, :] * np.log(self.initial_state_probabilities), axis=-1)

        # Term 2
        # np.log(self.transition_matrix) has shape (N, N)
        # The sum is over the second to last time points, all transitions, and all series
        term_2 = np.sum(self.v_hat * np.log(self.transition_matrix), axis=(-2, -1))

        # Term 3
        # self.densities has shape (K, T, N)
        # We take the log of self.densities and then multiply by self.u_hat
        # Term 3 for each series
        term_3 = np.sum(self.u_hat * np.log(final_densities), axis=(-2, -1))

        # Total Log-Likelihood
        LLN_series = term_1 + np.sum(term_2, axis=-1) + term_3
        return - np.sum(LLN_series)





























# ===============================================================
# |               Algorithm for the RSDC model                  |
# ===============================================================

class RSDCEM(Base):
    # Function Structure:
    '''
    init:                                   Setup
        df_to_array:                            prepare data from a dataframe, and save labels
        create_transition_matrix:               Create a transition matrix (2 x 2) for now
        set_univariate_density:                 Sets GARCH or ARMARCH residuals
        estimate_univariate_parameters          Runs Either GARCH or ARMACH for residuals
        set_parameters:                         Creates the initial rho parameters
        univariate_density:                     Calculate densities, should take t, parameters, state
        calculate_standard_deviations           Calculates the standard deviations of the univariate timeseries
        calculate_standardized_residuals        Calculates the standardized residuals if the univariate model.
        Density                                 Returns the probability density of the RSDC model
        log_densities                           As Density, but takes an argument for the matrix R, and returns log density. (For use in Minimize)
        form_correlation_matrix:                Takes array of rhos, splits into 2, creates correlation matrix for each, and RETURNS! R_matrices, det_R, inv_R, 
    fit:                                    Runs E_step, M_step & track
        E_step:                                 Runs forward & backward pass, and calculates smoothed probabilities
            get_denities:                           Runs form_correlation_matrix, & Create array of densities, dim = K, T, N        
            forward_pass:                           Calculate the forward probabilities, alpha
            backward_pass:                          Calculate the backward probabilities, beta
            calculate_smooothed_probabilities:      Use alpha & beta to calculate smoothed state probabilities & smoothed transition probabilities
        M_step:                                 Estimates total log likeihood and model parameters
            calculate_initial_states:               Calculate the initial state probabilities at time t=0
            estimate_model_parameters:              Minimize objective_function
            objective_funciton:                     Runs form_correlation_matrix, get_param_densities, and sums log_densities. Returns - sum
            get_param_densities:                    Runs log_densities, for t, n
            calculate_log_likelihood:               sets log_likelihood to self.result. fun (self.current_likelihood,)            
        track:                                  Manages parameter histories, and convergence.
    
    # finalize_parameters:                        Runs minimize with 'L-BFGS-B' on objective_function for final parameter estimates, conditional on states, and for standard deviations, errors etc.
    #     objective function:                         Turns flat array into matrix, then gets density array, and negative log likelihood.
    #     final_densities:                            Return Density based estimates 
    #     final_get_densities:                        Gets Density array for estimated densities
    #     final_log_likelihood:                       Use final_densities to calculate final log likelihood
    # results:

        

    #     summarize:                              gives estimated parameters, standard deviations etc.
    #     plot_smoothed_states:                   Plotting smoothed states.
    #     plot_reults:                            Plotting estimates
    #     plot_convergence:                       Plotting convergence
    #     plot_residuals:                         Plotting residuals
    #     plot_smoothed_history:                  Create a 3D plot of a smoothed state, for time interval t=0 to t=250
        

    '''
    def __init__(self, dataframe, univariate_parameters=None, squared=False, *args, **kwargs):
        super().__init__(dataframe, *args, **kwargs)
        # self.univariate_parameters = univariate_parameters
        self.squared = squared
        self.K_m = self.K # The number of the log likelihoods.
        
        self.K = 1
        if univariate_parameters is None:
            self.univariate_parameters = np.zeros((self.K_m, 3))  
            self.estimate_univariate_parameters()
        else:
            self.univariate_parameters = univariate_parameters

        self.correlation_parameters = int(self.K_m * (self.K_m - 1) / 2)
        self.num_parameters = self.correlation_parameters
        # Initialize parameters array
        self.set_parameters()
        self.calculate_standard_deviations()
        self.calculate_standardized_residuals()
        self.densities = np.zeros((self.K, self.T, self.n_states))
        self.log_likelihood_history = np.zeros(self.max_iterations)

    def set_parameters(self):
        # Assuming self.correlation_parameters is already defined correctly
        # Initialize rho as a numpy array with zeros for all states
        self.rho = np.zeros(self.correlation_parameters * self.n_states)  # Adjusted for a one-dimensional array
        
        # Set first half (corresponding to state 0) to 0.1
        first_half_end = self.correlation_parameters  # Calculate the end index for the first half
        self.rho[:first_half_end] = -0.3
        
        # Set second half (corresponding to state 1) to 0.0
        # This step might be redundant if the array is initialized with zeros, 
        # but it's included for clarity and in case the initialization value changes.
        second_half_start = first_half_end  # Calculate the start index for the second half
        self.rho[second_half_start:] = 0.3
        
        # Initialize parameter history
        self.parameter_history = np.zeros((self.max_iterations, self.correlation_parameters, self.n_states))
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

    def form_correlation_matrix(self, initial_parameters):
        # Takes Parameters, and forms a lost for each.
        # Calculate the number of parameters for each state's correlation matrix
        num_params_per_state = len(initial_parameters) // self.n_states
        
        # Initialize arrays to store the results
        R_matrix = np.zeros((self.n_states, self.K_m, self.K_m))
        det_R = np.zeros(self.n_states)
        inv_R = np.zeros((self.n_states, self.K_m, self.K_m))
        
        # Split the rhos into n_states equally long arrays
        for state in range(self.n_states):
            start_index = state * num_params_per_state
            end_index = start_index + num_params_per_state
            state_rhos = initial_parameters[start_index:end_index]
            
            # Calculate the indices for the upper triangular part excluding the diagonal
            triu_indices = np.triu_indices(self.K_m, 1)
            
            # Fill the correlation matrix for this state
            R_matrix[state][np.diag_indices(self.K_m)] = 1  # Diagonal elements are 1
            R_matrix[state][triu_indices] = state_rhos  # Upper triangular off-diagonal elements
            R_matrix[state][triu_indices[1], triu_indices[0]] = state_rhos  # Mirror to the lower triangular part
            
            # Compute determinant and inverse for each state's correlation matrix
            det_R[state] = np.linalg.det(R_matrix[state])
            inv_R[state] = np.linalg.inv(R_matrix[state])
    
        return R_matrix, det_R, inv_R


    def estimate_univariate_parameters(self):
        # Loop over each time series to estimate its parameters
        for k in range(self.K_m):
            # Extract the time series data
            data = self.data[k, :]
            
            # Optimize parameters using the appropriate density function
            if self.squared:  # GARCH model
                initial_guess = [0.1, 0.15, 0.6]  # Example initial guess for [omega, alpha, beta]
                result = minimize(self.negative_log_likelihood, initial_guess, args=(data, self.GARCH), method='L-BFGS-B', bounds=[(0.001, None), (0, 1), (0, 1)])
            else:  # ARMACH model
                initial_guess = [0.1, 0.15, 0.6]  # Example initial guess for [omega, alpha, beta]
                result = minimize(self.negative_log_likelihood, initial_guess, args=(data, self.ARMACH), method='L-BFGS-B', bounds=[(0.001, None), (0, 1), (0, 1)])
            
            # Store the optimized parameters
            if result.success:
                self.univariate_parameters[k, :] = result.x
            else:
                print(f"Parameter estimation failed for series {k}")


    def calculate_standard_deviations(self):
        # Preallocate sigma array with the shape of self.data
        sigmas = np.zeros_like(self.data)

        # Initial variance based on the historical data for each series
        initial_variances = np.var(self.data, axis=1)

        # Set initial variance for each series
        for k in range(self.K_m):
            sigmas[k, 0] = initial_variances[k]

        # Calculate sigmas for each time t using the appropriate model
        for t in range(1, self.T):
            for k in range(self.K_m):
                if self.squared:
                    # GARCH
                    sigmas[k, t] = self.univariate_parameters[k, 0] + self.univariate_parameters[k, 1] * self.data[k, t-1]**2 + self.univariate_parameters[k, 2] * sigmas[k, t-1]
                else:
                    # ARMACH
                    sigmas[k, t] = self.univariate_parameters[k, 0] + self.univariate_parameters[k, 1] * np.abs(self.data[k, t-1]) + self.univariate_parameters[k, 2] * np.abs(sigmas[k, t-1])

        # If squared=False, take the square root for GARCH standard deviations
        if self.squared:
            sigmas = np.sqrt(sigmas)

        self.standard_deviations = sigmas


    def calculate_standardized_residuals(self):
        # The original method may have inaccuracies in inverting and multiplying matrices.
        # Correct approach for element-wise division to get standardized residuals:
        self.residuals = self.data / self.standard_deviations

    # def form_correlation_matrix(self):
    #     # Assuming self.rho is a flat array containing all parameters for all states
    #     assert len(self.rho) == self.n_states * self.num_parameters, "Mismatch in the number of parameters and expected size"

    #     # Initialize arrays to store the results
    #     R_matrix = np.zeros((self.n_states, self.K_m, self.K_m))
    #     det_R = np.zeros(self.n_states)
    #     inv_R = np.zeros_like(R_matrix)

    #     # Process each state
    #     for state in range(self.n_states):
    #         # Calculate start and end index for the parameters of this state in self.rho
    #         start_idx = state * self.num_parameters
    #         end_idx = start_idx + self.num_parameters
    #         state_params = self.rho[start_idx:end_idx]

    #         # Fill the off-diagonal elements of the R matrix for this state
    #         R_matrix[state][np.diag_indices(self.K_m)] = 1  # Diagonal elements are 1
    #         triu_indices = np.triu_indices(self.K_m, 1)
    #         tril_indices = (triu_indices[1], triu_indices[0])  # Swap rows and cols for the lower triangle

    #         # Assuming state_params can fill the upper triangle off-diagonal
    #         R_matrix[state][triu_indices] = state_params
    #         R_matrix[state][tril_indices] = state_params  # Symmetric lower part

    #         # Compute determinant and inverse for this state's R matrix
    #         det_R[state] = np.linalg.det(R_matrix[state])
    #         inv_R[state] = np.linalg.inv(R_matrix[state])

    #     self.R_matrix = R_matrix
    #     self.det_R = det_R
    #     self.inv_R = inv_R


    def get_densities(self):
        # Get correlation matrix in each state
        self.R_matrix, self.det_R, self.inv_R = self.form_correlation_matrix(self.rho)


        for n in range(self.n_states):
            for t in range(self.T):
                self.densities[:,t, n] = self.Density(t,n)
                # self.densities[:, t, n] = self.Density(t,n)




    def Density(self, t, n):
        term_1 = self.K_m * np.log(2 * np.pi)
        D = np.diag(self.standard_deviations[:,t])
        det_D = np.linalg.det(D)
        det_D = np.max((det_D, 1e-9))
        term_2 = 2 * np.log(det_D)  # Assuming standard_deviations is calculated
        term_3 = np.log(self.det_R[n] + 1e-8)  # Ensure numerical stability
        term_4 = self.residuals[:, t].T @ self.inv_R[n] @ self.residuals[:, t]

        # Calculate the log likelihood for each k and t, stored as densities for visualization
        return np.exp(-0.5 * (term_1 + term_2 + term_3 + term_4))


    def estimate_model_parameters(self):
        # Calculate the number of parameters needed to form the correlation matrix R
        num_parameters = self.K_m * (self.K_m - 1) / 2

        # Define initial_parameters
        # For simplicity, starting with all parameters set to a small value close to 0,
        # indicating initial low correlation
        initial_parameters = self.rho
        # constraints = {'type': 'ineq', 'fun': self.optimization_constraint}
        bounds = [(-0.99, 0.99) for _ in range(int(num_parameters * 2))]
        # self.calculate_standard_deviations()
        # self.calculate_standardized_residuals()

        #options={}
        # During optimization, use a constraint
        def objective_function(initial_parameters):
            return self.objective_function_vectorized(initial_parameters)


        self.result = minimize(
            objective_function,  # Your objective function
            initial_parameters,  # Initial guess of the parameters
            method='TNC',
            # constraints=constraints,
            # options=options,
            bounds=bounds
        )
        #print(self.result.x)
        self.rho = self.result.x
        self.model_parameters = self.result.x
        self.R_matrix = self.form_correlation_matrix(self.rho)
        self.current_likelihood = self.result.fun

    def objective_function_vectorized(self, initial_parameters):
        log_densities = self.get_param_densities(initial_parameters)

        # Term 1
        term_1 = np.sum(self.u_hat[:, 0] * np.log(self.initial_state_probabilities + 1e-8), axis=1)

        # Term 2
        log_transition_matrix = np.log(self.transition_matrix + 1e-8)
        term_2 = np.sum(self.v_hat * log_transition_matrix[np.newaxis, np.newaxis, :, :], axis=(1, 2, 3))

        # Term 3
        term_3 = np.sum(self.u_hat * log_densities, axis=(1, 2))

        # Combine terms to calculate the total Log-Likelihood for each series
        LLN_series = term_1 + term_2 + term_3

        return -LLN_series



    def get_param_densities_vectorized(self, initial_parameters):
        R_matrix, R_det, R_inv = self.form_correlation_matrix_vectorized(initial_parameters)

        # Preparing for vectorized computation of log densities
        # Assuming self.residuals is shaped (K_m, T), with K_m as the number of variables
        term_1 = -0.5 * self.K_m * np.log(2 * np.pi)
        
        # Vectorized computation of determinant and inverse of D (diagonal matrix of standard deviations)
        log_det_D = np.log(self.standard_deviations**2).sum(axis=0)  # Assuming standard deviations are (K_m, T)
        term_2 = -0.5 * log_det_D
        
        # Term 3: Log determinant of R
        term_3 = -0.5 * np.log(R_det + 1e-8)[:, np.newaxis]  # Shape (n_states, 1) for broadcasting
        
        # Term 4: Quadratic term
        # Reshaping residuals for broadcasting: New shape (K_m, 1, T)
        residuals_reshaped = self.residuals[:, np.newaxis, :]
        # R_inv is (n_states, K_m, K_m), need to perform batch matrix multiplication with residuals
        # Transpose residuals to align with the operation: (T, K_m)
        residuals_transposed = self.residuals.T

        # Execute the quadratic form operation for each state and time point
        # New einsum path: 'nij,tj,ti->nt'
        # Note: 'ti->nt' might seem counterintuitive, but it correctly represents the operation intended
        term_4 = -0.5 * np.einsum('nij,tj,ti->nt', R_inv, residuals_transposed, residuals_transposed)
        
        # Summing terms to get log densities: Shape will be (n_states, T)
        log_densities = term_1 + term_2 + term_3 + term_4
        
        # Transpose to match expected shape: (1, T, n_states)
        return log_densities.transpose((2, 1, 0))























    def objective_function(self, initial_parameters):

        log_densities = self.get_param_densities_vectorized(initial_parameters)

        """
        The log likelihood is the sum of 3 terms.
        log_likelihood = term_1 + term_2 + term_3
        term_1 = sum_{j=1}^n_states         u_hat[state, t=1] * np.log(initial_state_probability[j]) 

        term_2 = sum_{j=1}^n_states
                    sum_{j=1}^n_states 
                        sum_{t=1}^T        v_hat[j,k,t] * log gamma[jk]
        term_3 = sum_{j=1}^n_states 
                        sum_{t=1}^T        u_hat[j,k,t] * log densitiy_j[t]
        """
        # Term 1
        # self.u_hat[:, 0, :] has shape (K, N) representing the probability of being in each state at t=1 for all series
        # np.log(self.initial_state_probabilities) has shape (N,) representing the log of the initial state probabilities
        term_1 = np.sum(self.u_hat[:, 0, :] * np.log(self.initial_state_probabilities + 1e-8), axis=-1)


        # Term 2
        # np.log(self.transition_matrix) has shape (N, N)
        # The sum is over the second to last time points, all transitions, and all series
        term_2 = np.sum(self.v_hat * np.log(self.transition_matrix), axis=(-2, -1))

        # Term 3
        # self.densities has shape (K, T, N)
        # We take the log of self.densities and then multiply by self.u_hat
        # Term 3 for each series
        term_3 = np.sum(self.u_hat * log_densities,)# axis=(-2, -1))
        # Total Log-Likelihood

        LLN_series = term_1 + np.sum(term_2, axis=-1) + term_3

        return - LLN_series


    def get_param_densities(self, initial_parameters):
        # Get correlation matrix in each state
        param_densities = np.zeros((self.K, self.T, self.n_states))
        # log_densities = np.zeros((self.K, self.T, self.n_states))
        R_matrix, R_det, R_inv = self.form_correlation_matrix(initial_parameters)
        # print(R_matrix)#\nThe determinant is: {R_det}')
        # print(f'The matrix is: {R_matrix}')#\nThe determinant is: {R_det}')


        for n in range(self.n_states):
            if R_det[n]<1e-9:
                R_det[n] = 1e-9
            for t in range(self.T):
                param_densities[0, t, n] = self.log_density(t, n, R_det, R_inv)
        return param_densities
               

    def log_density(self, t, n, R_det, R_inv):
        term_1 = self.K_m * np.log(2 * np.pi)
        D = np.diag(self.standard_deviations[:,t])
        det_D = np.linalg.det(D)
        det_D = np.max((det_D, 1e-9))
        term_2 = 2 * np.log(det_D)  # Assuming standard_deviations is calculated
        term_3 = np.log(R_det[n])  # Ensure numerical stability
        term_4 = self.residuals[:, t].T @ R_inv[n] @ self.residuals[:, t]

        # Calculate the log likelihood for each k and t, stored as densities for visualzation
        return -0.5 * (term_1 + term_2 + term_3 + term_4)

     
 
 
    def is_positive_semi_definite(self, R):
        # Check if all eigenvalues are non-negative
        eigenvalues, _ = np.linalg.eig(R)
        return np.all(eigenvalues >= -1e-8)  # A small tolerance for numerical stability

    def optimization_constraint(self, parameters):
        # This function needs to return a value greater than or equal to 0 for feasible solutions
        R = self.parameters_to_correlation_matrix(parameters)  # Assume this function returns R without setting it on self
        if self.is_positive_semi_definite(R):
            return 1.0  # Arbitrary positive value to indicate a feasible solution
        else:
            return -1.0  # Indicates an infeasible solution
    

    def calculate_log_likelihood(self):
        pass




    def track(self, iteration, ):
        # Log-Likelihood History
        self.log_likelihood_history[iteration] = self.current_likelihood
        # np.zeros((self.max_iterations, self.K))
        self.parameter_history[iteration] = self.model_parameters.copy()
        # parameter_history(self.max_iterations, 3 (mu, phi, sigma), self.K, self.n_states)
        self.initial_state_probabilities_history[iteration] = self.initial_state_probabilities.copy()
        self.transition_matrix_history[iteration] = self.transition_matrix.copy()