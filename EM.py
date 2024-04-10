import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import time

from tqdm import tqdm
from scipy.optimize import minimize

class Base(object):
    def __init__(self, dataframe, n_states=2, max_iterations=200, tolerance=1e-6):
        # Extract labels and data array from dataframe
        # Data is structured in the following way: K_series, T observations, N states 
        self.dataframe = dataframe
        self.data, self.labels = self.df_to_array(self.dataframe)

        # set number of states
        self.n_states = n_states

        # Get data dimensions:
        self.K, self.T = self.data.shape
      
        # Setup probabilities
        self.p_00, self.p_11 = 0.995, 0.995
        self.transition_matrix = self.create_transition_matrix(self.p_00, self.p_11) 
        self.delta = np.ones(self.n_states) / self.n_states

        # Model Settings
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        # Setup Model Parameters
        self.mu = np.zeros((self.K, self.n_states))
        self.phi = np.zeros((self.K, self.n_states))
        self.sigma = np.zeros((self.K, self.n_states))
        self.set_initial_parameters()

        # Missing:
        self.densities = np.zeros((self.K, self.T, self.n_states))

    def df_to_array(self, dataframe):
        # Create Numpy Array
        data_array = dataframe.to_numpy().T

        # Get Column Labels
        labels = dataframe.columns.tolist()

        return data_array, labels


    def create_transition_matrix(self, p_00, p_11):
        transition_matrix = np.zeros([2,2])

        transition_matrix[0] = p_00, 1 - p_11
        transition_matrix[1] = 1 - p_00, p_11

        transition_matrix = transition_matrix
        # Return the Transition Matrix
        return transition_matrix

    def set_initial_parameters(self):
        for k in range(self.K):
            mean_k = np.mean(self.data[k, :])
            std_k = np.sqrt(np.var(self.data[k, :]))
            
            for state in range(self.n_states):
                # Perturb mean and standard deviation for diversity across states
                self.mu[k, state] = mean_k + (state - self.n_states / 2) * 0.01
                self.sigma[k, state] = std_k + (state - self.n_states / 2) * 0.1


    def gaussian_pdf(self, x, x_1, mu, phi, sigma):
        """
        Compute the Gaussian PDF manually for an array of observations.
        
        Parameters:
        - x: An array of observations.
        - mu: The mean of the Gaussian distribution.
        - sigma: The standard deviation of the Gaussian distribution.
        
        Returns:
        - The PDF values for each observation in x.
        """
        # Ensure sigma is positive
        return (1. / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu - phi * x_1) / sigma) ** 2) + 1e-6

    def get_densities(self):
        """
        Compute the likelihood of each observation under each state's Gaussian distribution for K time series.
        
        Parameters:
        - data: A 3D array (K, N, T) of observations, where K is the number of time series, 
                N is the number of variables, and T is the number of time steps.
        - mu: An array of means for each state's Gaussian distribution.
        - sigma: An array of standard deviations for each state's Gaussian distribution.
        
        Returns:
        - A 3D array (K, T, n_states) of likelihoods of observations for each state across each time series.
        """
        for k in range(self.K):
            for state in range(self.n_states):
                for t in range(self.T):
                    self.densities[k, t, state] = self.gaussian_pdf(self.data[k, t], 
                        self.data[k, t-1], 
                        self.mu[k, state], 
                        self.phi[k,state],
                        self.sigma[k, state])
                    


    def forward_pass(self,):
        # Initialize the forward probability matrix alpha with shape (K, T, n_states)
        # and a scaling factor array with shape (K, T)
        alpha = np.zeros((self.K, self.T, self.n_states))
        scale_factors = np.zeros((self.K, self.T))
        
        # Initial step: Compute initial alpha values and scale them
        alpha[:, 0, :] = self.delta * self.densities[:, 0, :]
        scale_factors[:, 0] = 1.0 / np.sum(alpha[:, 0, :], axis=1)
        alpha[:, 0, :] *= scale_factors[:, 0, np.newaxis]
        
        # Recursive step: Update alpha values for t > 0
        for t in range(1, self.T):
            # Vectorized update of alpha using matrix multiplication for transition probabilities
            for k in range(self.K):
                alpha[k, t, :] = np.dot(alpha[k, t-1, :], self.transition_matrix) * self.densities[k, t, :]
            
            # Scale alpha values to prevent underflow
            scale_factors[:, t] = 1.0 / np.sum(alpha[:, t, :], axis=1)
            alpha[:, t, :] *= scale_factors[:, t, np.newaxis]
        
        self.alpha = alpha
        self.scale_factors = scale_factors


    def backward_pass(self,):
        # Initialize the backward probability matrix beta with shape (K, T, n_states)
        beta = np.zeros((self.K, self.T, self.n_states))
        
        # Initial step: Set the last beta values to 1 and scale them by the last scale factors
        beta[:, self.T-1, :] = 1.0 * self.scale_factors[:, self.T-1, np.newaxis]
        
        # Recursive step: Update beta values from T-2 down to 0
        for t in range(self.T-2, -1, -1):
            for k in range(self.K):
                # Vectorized update of beta using matrix multiplication for transition probabilities
                # Note: Need to multiply by the densities for time t+1
                beta[k, t, :] = np.dot(self.transition_matrix, (self.densities[k, t+1, :] * beta[k, t+1, :]))
            
            # Scale beta values to prevent underflow, using the same scale factors as forward pass
            beta[:, t, :] *= self.scale_factors[:, t, np.newaxis]
        
        self.beta = beta

    def calculate_smoothed_probabilities(self):
        # Smoothed state probabilities
        numerator = self.alpha * self.beta  # Element-wise multiplication
        denominator = numerator.sum(axis=2, keepdims=True)  # Sum over states for normalization
        self.u_hat = numerator / denominator
        
        # Precompute for smoothed transitions to avoid recomputation
        alpha_shifted = np.roll(self.alpha, shift=1, axis=1)  # Shift alpha by 1 to align t-1 with t
        alpha_shifted[:, 0, :] = 0  # Set initial shifted values to 0 as there's no t-1 for the first element
        
        # Smoothed transition probabilities
        numerator_v = alpha_shifted[:, :-1, :, np.newaxis] * self.transition_matrix[np.newaxis, np.newaxis, :, :] * self.densities[:, 1:, np.newaxis, :] * self.beta[:, 1:, np.newaxis, :]
        denominator_v = numerator_v.sum(axis=(2, 3), keepdims=True) + 1e-6 # Sum over both j and k states for normalization
        self.v_hat = numerator_v / denominator_v
    
    def update_initial_state_probabilities(self):
        # Calculate the mean of the smoothed state probabilities at t=0 across all K series
        initial_state_probabilities = self.u_hat[:, 0, :].mean(axis=0)
        self.delta = initial_state_probabilities


    def estimate_state_parameters(self):
        # Adjust parameter matrices to have self.K rows and self.n_states columns
        self.mu_hat = np.zeros((self.K, self.n_states))
        self.sigma_hat = np.zeros((self.K, self.n_states))
        self.phi_hat = np.zeros((self.K, self.n_states))
        
        # Iterate over each state to calculate parameters
        for j in range(self.n_states):
            # Weighted mean and variance for each series
            for k in range(self.K):
                u_hat_jk = self.u_hat[k, :, j]  # Smoothed probabilities for state j in series k
                
                # Calculate the weighted mean for state j in series k
                weighted_sum = np.sum(u_hat_jk * self.data[k, :])
                total_weight = np.sum(u_hat_jk)
                self.mu_hat[k, j] = weighted_sum / total_weight
                
                # Calculate the weighted variance for state j in series k
                weighted_variance_sum = np.sum(u_hat_jk * (self.data[k, :] - self.mu_hat[k, j])**2)
                self.sigma_hat[k, j] = weighted_variance_sum / total_weight
                
                # Correcting AR(1) parameter estimation
                if total_weight > 1:  # Ensure there's enough data for estimation
                    X = self.data[k, :-1]  # Observations at t-1
                    Y = self.data[k, 1:]  # Observations at t
                    u_hat_jk_shifted = u_hat_jk[1:]  # Shifted to align with X and Y
                    
                    # Compute elements for AR(1) parameter estimation
                    phi_numerator = np.sum(u_hat_jk_shifted * X * Y) - np.sum(u_hat_jk_shifted * X) * np.sum(u_hat_jk_shifted * Y) / np.sum(u_hat_jk_shifted)
                    phi_denominator = np.sum(u_hat_jk_shifted * X**2) - (np.sum(u_hat_jk_shifted * X) ** 2) / np.sum(u_hat_jk_shifted)
                    
                    self.phi_hat[k, j] = phi_numerator / phi_denominator if phi_denominator != 0 else 0

    def calculate_log_likelihood(self):
        # Assuming self.alpha is the forward probabilities with shape (K, T, n_states)
        # and self.scale_factors is used for numerical stability
        log_likelihoods = np.log(self.scale_factors).sum(axis=1)  # Sum log scale factors for each series
        total_log_likelihood = log_likelihoods.sum()  # Sum across all series for total log likelihood
        return total_log_likelihood

    def switch_states_if_necessary(self):
        for k in range(1, self.K):  # Start from the second series since we compare everything to the first one
            # Calculate the differences as described
            diff_state_0 = np.abs(self.u_hat[k, :, 0] - self.u_hat[0, :, 0]).mean()
            diff_state_1 = np.abs(self.u_hat[k, :, 1] - self.u_hat[0, :, 0]).mean()

            # Check if switching states results in a smaller difference
            if diff_state_1 < diff_state_0:
                # Switch states for u_hat for series k
                self.u_hat[k, :, [0, 1]] = self.u_hat[k, :, [1, 0]]

                # Since we switched states, we should switch the parameters as well
                # Switch mu, phi, sigma for series k
                self.mu[k, [0, 1]] = self.mu[k, [1, 0]]
                self.phi[k, [0, 1]] = self.phi[k, [1, 0]]
                self.sigma[k, [0, 1]] = self.sigma[k, [1, 0]]
 


    def fit(self):
        #self.log_likelihood_history = []  # Initialize a list to store log likelihood history

        for iteration in tqdm(range(self.max_iterations), desc='Fitting Model'):
            # Your existing fitting code

            # Get Densities
            self.get_densities()

            # Forward Pass
            self.forward_pass()

            # Backward Pass
            self.backward_pass()

            # Smoothing
            self.calculate_smoothed_probabilities()
            self.update_initial_state_probabilities()

            # self.aggregate_smoothed_probabilities()  # Ensure common states across time series
            # self.m_step()
            # Before re-estimating parameters, ensure consistency in states
            self.switch_states_if_necessary()
            # Parameters
            self.estimate_state_parameters()

            # Set Transition Probabilities
            # self.update_transition_and_initial_parameters()

            # Set Model Parameters            
            self.mu = self.mu_hat
            self.phi = self.phi_hat
            self.sigma = self.sigma_hat
            self.update_transition_and_initial_parameters()
            # Calculate and store the log likelihood
            # current_log_likelihood = self.calculate_log_likelihood()
            # self.log_likelihood_history.append(current_log_likelihood)

            # Optional: Check for convergence based on the change in log likelihood
            # if iteration > 0:
            #     likelihood_change = abs(self.log_likelihood_history[-1] - self.log_likelihood_history[-2])
            #     if likelihood_change < self.tolerance:
            #         print(f"Convergence achieved after {iteration+1} iterations.")
            #         break







    def update_transition_and_initial_parameters(self):
        # Aggregate smoothed probabilities and transitions over K series if needed
        # Assuming u_hat and v_hat shapes are now (T, n_states) and (T-1, n_states, n_states) after aggregation
        u_hat_aggregated = self.u_hat.mean(axis=0)  # If multiple series, otherwise directly use self.u_hat
        v_hat_aggregated = self.v_hat.mean(axis=0)  # Same as above
        
        # Update transition matrix parameters
        # Sum v_hat over time for numerator, and u_hat over time (excluding last time step) for denominator
        numerator = v_hat_aggregated.sum(axis=0)
        denominator = u_hat_aggregated[:-1].sum(axis=0)[:, np.newaxis]  # Reshape for broadcasting
        transition_matrix = numerator / denominator
        print(transition_matrix)
        self.transition_matrix = transition_matrix
        # Normalize the transition matrix rows to sum to 1
        # row_sums = self.transition_matrix.sum(axis=1, keepdims=True)
        # self.transition_matrix /= row_sums
            
        # Update initial state probabilities
        # Directly use the first time step's smoothed state probabilities
        self.delta = u_hat_aggregated[0, :]



    def update_transition_and_initial_parameters(self):
        # Use smoothed probabilities and transitions from the first series only
        # Assuming u_hat shape is (K, T, n_states) and v_hat shape is (K, T-1, n_states, n_states)
        # Select the first series by indexing with 0
        u_hat_first_series = self.u_hat[0]  # Shape (T, n_states)
        v_hat_first_series = self.v_hat[0]  # Shape (T-1, n_states, n_states)

        # Update transition matrix parameters
        # Sum v_hat over time for numerator and u_hat over time (excluding last time step) for denominator
        numerator = v_hat_first_series.sum(axis=0)
        denominator = u_hat_first_series[:-1].sum(axis=0)  # Reshape for broadcasting is not needed as it's already 1D

        # Prevent division by zero by adding a small value to the denominator
        denominator_safe = np.where(denominator == 0, 1e-6, denominator)
        new_transition_matrix = numerator / denominator_safe

        # Ensure the transition matrix rows sum to 1
        row_sums = new_transition_matrix.sum(axis=1, keepdims=True)
        new_transition_matrix = np.where(row_sums == 0, 0, new_transition_matrix / row_sums)

        self.transition_matrix = new_transition_matrix

        # Update initial state probabilities using the first time step's smoothed state probabilities from the first series
        self.delta = u_hat_first_series[0, :]

    def m_step(self):
        """
        M-step: Numerically optimize the transition matrix and initial state probabilities
        using the aggregated smoothed state and transition probabilities.
        """
        
        def negative_log_likelihood(params):
            """
            Calculate the negative log likelihood of the observed data given the model parameters.
            This function will be minimized to find the optimal parameters.
            """
            # Unpack parameters: transition probabilities and initial state probabilities
            p_00, p_11 = params[:2]
            delta = params[2:]
            
            # Enforce constraints (e.g., probabilities sum to 1)
            p_01 = 1 - p_00
            p_10 = 1 - p_11
            
            # Update model with new parameters
            self.transition_matrix = np.array([[p_00, p_01], [p_10, p_11]])
            self.delta = delta
            
            # Recalculate densities, forward and backward passes, and smoothed probabilities
            self.get_densities()
            self.forward_pass()
            self.backward_pass()
            self.calculate_smoothed_probabilities()
            
            # Calculate and return the negative log likelihood
            # Note: Define the log_likelihood function based on alpha, beta, and the observations
            return -self.calculate_log_likelihood()
        
        # Initial guess for parameters: existing transition probabilities and initial state probabilities
        initial_guess = [self.transition_matrix[0, 0], self.transition_matrix[1, 1]] + list(self.delta)
        
        # Constraints and bounds to ensure the parameters remain valid probabilities
        constraints = ({'type': 'eq', 'fun': lambda x: x[0] + x[1] - 1})  # Example constraint, adapt as needed
        bounds = [(0, 1) for _ in initial_guess]  # Probability bounds
        
        # Perform numerical optimization
        result = minimize(negative_log_likelihood, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        
        # Update model parameters with optimized values
        p_00, p_11 = result.x[:2]
        self.delta = result.x[2:]
        self.transition_matrix = np.array([[p_00, 1 - p_11], [1 - p_00, p_11]])

    # def aggregate_smoothed_probabilities(self):
    #     """
    #     Aggregate smoothed state and transition probabilities across all series.
    #     """
    #     self.u_hat_aggregated = self.u_hat.mean(axis=0)
    #     self.v_hat_aggregated = self.v_hat.mean(axis=0)










    # class RSDC(Base):
    #     """docstring for RSDC"""
    #     def __init__(self, arg):
    #         super(RSDC, self).__init__()
    #         self.arg = arg
    #         