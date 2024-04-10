# =================================================================================
# |         For Estimating Hidden Markov Models & Markov Switching Models         |
# =================================================================================
import numpy as np
import pandas as pd
import time

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.optimize import minimize
import yfinance as yf

from arch import arch_model
from tqdm import tqdm
from scipy.optimize import minimize
from scipy.stats import t, norm, skew, kurtosis
from scipy.stats.mstats import gmean

import numdifftools as nd
import matplotlib.gridspec as gridspec  # For advanced subplot layout

# ===========================================================
# |                     Base Model                          |
# ===========================================================
class Base:
    def __init__(self, dataframe, transition_guess=0.99, n_states=2, max_iterations=100, tolerance=1e-5, num_cal=10, univariate_parameters=None):

        # Basic Model Settings
        self.max_iterations = max_iterations
        self.tolerance = tolerance       
        self.dataframe = dataframe

        # Model states
        self.n_states = n_states

        # Transform dataframe to numpy array & Get Labels
        self.data, self.labels = self.df_to_array(self.dataframe)
        self.univariate_parameters = univariate_parameters
        # Data Dimensions 
        self.K, self.T = self.data.shape

        # Set how often to use numeric Estimatinon
        self.num_cal = num_cal
        # Set Estimation Dimensions
            # 1 for a single output. Examples: univariate timeseries, RSDC, VAR,
            # self.K for multiple univariate timesereis under the same regime. 

       
        # Parameter, probability and Likelihood Histories
        self.probability_history = np.zeros((self.max_iterations+1, self.n_states, self.n_states)) # Max_iterations + 1, n_states ** 2, n_states
        self.log_likelihood_history = np.zeros((self.max_iterations+1)) # max_iterations + 1, Estimation Dimensions
        self.initial_state_history = np.zeros((self.max_iterations+1, self.n_states))

        # Save initial Histories
        # self.parameter_history[0:,:] = 
        # self.probability_history[0:,:] = 
        # self.log_likelihood_history[0:] = 
        #initial_state_probabilities_history


        # Create array for probability densities of the model
        self.densities = np.zeros((self.T, self.n_states))

        self.setup(transition_guess)


    def df_to_array(self, dataframe):
        # Create Numpy Array
        data_array = dataframe.to_numpy().T

        # Get Column Labels
        labels = dataframe.columns.tolist()

        return data_array, labels


    def setup(self, transition_guess):
        # Calculate Univariate if none
        if self.univariate_parameters is None:
            self.estimate_univariate()

        # Form Transition Matrix
        self.transition_matrix = self.form_transition_matrix(transition_guess)
        
        # Set Initial State Probability to 1/n_states
        self.initial_states = np.ones(self.n_states) / self.n_states

        # Set parameters & settings specific to the model
        self.num_parameters = int(self.K * (self.K - 1) / 2)
        # self.set_parameters()

        self.total_iterations = 0
        self.standard_deviations = self.calculate_standard_deviations()
        self.residuals = self.calculate_residuals()
        self.correlation_matrix = self.generate_initial_correlation_matrices()
        self.parameter_history = np.zeros((self.max_iterations+1, self.n_states, self.num_parameters)) # Max_iterations + 1, number of parameters, n_states
        for i in range(self.n_states):
            self.parameter_history[0,i,:] = self.get_upper_off_diagonal_elements(self.correlation_matrix[i,:,:])
        self.probability_history[0,:,:] = self.transition_matrix
        self.initial_state_history[0,:] = self.initial_states

    def estimate_univariate(self):
        # Initialize an array to store the omega, alpha, and beta parameters for each series
        params_array = np.zeros((self.K, 3))
        
        for k in range(self.K):
            # Select the k-th series from the data
            series = self.data[k, :]
            
            # Fit a GARCH(1,1) model to the series
            model = arch_model(series, vol='Garch', p=1, q=1)
            model_fit = model.fit(disp='off') # Set disp='off' to avoid printing the fit summary
            
            # Extract the parameters: omega (constant), alpha (ARCH), beta (GARCH)
            omega = model_fit.params['omega']
            alpha = model_fit.params['alpha[1]']
            beta = model_fit.params['beta[1]']
            
            # Store the parameters in the params_array
            params_array[k, :] = [omega, alpha, beta]
            
        self.univariate_parameters = params_array
        

    def form_transition_matrix(self, transition_guess):
        diagonal = transition_guess

        transition_matrix = diagonal* np.eye(self.n_states) + (1-diagonal) * (np.ones((self.n_states,self.n_states)) - np.eye(self.n_states,self.n_states)) / (self.n_states - 1)
        return transition_matrix

    def calculate_standard_deviations(self):
        sigmas = np.zeros((self.K, self.T))
        sigmas[:,0] = np.var(self.data, axis=1)

        for t in range(1, self.T):
            sigmas[:,t] = self.univariate_parameters[:,0] + self.univariate_parameters[:, 1] * self.data[:, t-1]**2 + self.univariate_parameters[:, 2] * sigmas[:, t-1] 
        
        sigma = np.sqrt(sigmas)
        return sigma


    def calculate_residuals(self):
        residuals = self.data / self.standard_deviations
        return residuals


    def generate_initial_correlation_matrices(self):
        # Create an array to hold the correlation matrices
        correlation_matrices = np.zeros((self.n_states, self.K, self.K))
        
        # Define the pattern of off-diagonal values based on N
        off_diagonal_values = np.linspace(-0.1, 0.1 * self.n_states, self.n_states)
        
        for i in range(self.n_states):
            # Fill the diagonal with 1s
            np.fill_diagonal(correlation_matrices[i], 1)
            
            # Fill the off-diagonal elements with the specified pattern
            np.fill_diagonal(correlation_matrices[i, :, 1:], off_diagonal_values[i])
            np.fill_diagonal(correlation_matrices[i, 1:, :], off_diagonal_values[i])
            
            # Ensure the matrix is symmetric
            correlation_matrices[i] = (correlation_matrices[i] + correlation_matrices[i].T) / 2

        for i in range(self.n_states):
            correlation_matrices[i] = self.cholesky_scale(correlation_matrices[i])
        # print(correlation_matrices)
        return correlation_matrices

    def cholesky_scale(self, matrix):
        # P = np.linalg.cholesky(matrix)
        # for j in range(K):
        #     # Compute the sum of squares of off-diagonal elements up to the diagonal
        #     sum_of_squares = np.sum(P[j, :j] ** 2)
        #     argument = max(1 - sum_of_squares, 0)
        #     # Adjust the diagonal element
        #     P[j, j] = np.sqrt(argument) if j > 0 else 1# P[j,j]
        # No need to recompute the matrix as P is already the adjusted Cholesky decomposition
        # return P

        P = np.linalg.cholesky(matrix)
        # print(f'P matrix: \n {P}')
        for j in range(self.K):
            sum = np.sum(P[j, :j] ** 2)
            P[j,j] = np.sqrt(1-sum) if 1 - sum > 0 else 0
            if j==0:
                P[j,j] = 1
        scaled = np.dot(P, P.T)
        # print(f'matrix Matrix \n{matrix }')
        # print(f'scaled Matrix \n{scaled }')
        return scaled
        

    def get_upper_off_diagonal_elements(self, matrix):
        """
        Extract the upper off-diagonal elements of a matrix and return them as a flat array.
        
        Parameters:
        - matrix (np.array): A square NumPy array from which to extract the upper off-diagonal elements.
        
        Returns:
        - np.array: A flat array containing the upper off-diagonal elements of the input matrix.
        """
        # Ensure the input is a NumPy array
        matrix = np.asarray(matrix)
        
        # Check if the matrix is square
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Matrix must be square.")
        
        # Get the indices for the upper triangle, starting from the first element above the diagonal
        i, j = np.triu_indices(n=matrix.shape[0], k=1)
        
        # Extract and return the elements using these indices
        return matrix[i, j]    

    def fit(self):# , data, univariate_parameters, num_states, max_it = 200, tol = 1e-7):
        # initial_states, standard_deviations, residuals, normalized_residuals, correlation_matrix, transition_matrix= setup(N, data, univariate_parameters)

        for i in range(self.max_iterations):
            # E step
            self.e_step()

            # M step
            self.m_step()
            # print(current_likelihood)
            # print(self.correlation_matrix)

            # Track
            self.log_likelihood_history[i + 1] = self.current_likelihood
            self.probability_history[i + 1] = self.transition_matrix
            for j in range(self.n_states):
                self.parameter_history[i + 1,j] = self.get_upper_off_diagonal_elements(self.correlation_matrix[j,:,:])
            self.initial_state_history[i + 1] = self.initial_states
            self.total_iterations += 1
            # Break
            if i > 1 and np.abs(self.log_likelihood_history[i] - self.log_likelihood_history[i-1]) < self.tolerance:
                # print(f'Model Converged at iteration:  {i}')
                break
    def flashy_fit(self):# , data, univariate_parameters, num_states, max_it = 200, tol = 1e-7):
        # initial_states, standard_deviations, residuals, normalized_residuals, correlation_matrix, transition_matrix= setup(N, data, univariate_parameters)

        for i in tqdm(range(self.max_iterations)):
            # E step
            self.e_step()

            # M step
            self.m_step()
            # print(current_likelihood)
            # print(self.correlation_matrix)

            # Track
            self.log_likelihood_history[i + 1] = self.current_likelihood
            self.probability_history[i + 1] = self.transition_matrix
            for j in range(self.n_states):
                self.parameter_history[i + 1,j] = self.get_upper_off_diagonal_elements(self.correlation_matrix[j,:,:])
            self.initial_state_history[i + 1] = self.initial_states
            self.total_iterations += 1
            # Break
            if i > 1 and np.abs(self.log_likelihood_history[i] - self.log_likelihood_history[i-1]) < self.tolerance:
                print(f'Model Converged at iteration:  {i}')
                break

        
        # self.correlation_matrix = correlation_matrix
        # self.transition_matrix = transition_matrix
        # self.log_likelihood_history = log_likelihood_history
        # self.u_hat = u_hat
        # self.standard_deviations = standard_deviations

    def num_fit(self):# , data, univariate_parameters, num_states, max_it = 200, tol = 1e-7):
        # initial_states, standard_deviations, residuals, normalized_residuals, correlation_matrix, transition_matrix= setup(N, data, univariate_parameters)
        # Setup your tqdm progress bar instance here
        pbar = tqdm(range(self.max_iterations), desc='Expectation Maximization')

        for i in pbar:
            # E step
            self.e_step()

            # M step
            self.m_step()
            # print(current_likelihood)
            # print(self.correlation_matrix)
            if i % self.num_cal == 0:
                pbar.set_description(desc='Optimizing Numerically', refresh=True)

                nan_mask = np.isnan(self.correlation_matrix)
                # Replace these NaN values with the specified replacement value
                self.correlation_matrix[nan_mask] = 0.1
                
                self.correlation_matrix = self.num_corr()
            # Since the condition for this update is not specified,
            # you may want to update it outside the if statement or under certain conditions.
            pbar.set_description('Expectation Maximization', refresh=True)

            self.log_likelihood_history[i + 1] = self.current_likelihood
            self.probability_history[i + 1] = self.transition_matrix
            for j in range(self.n_states):
                self.parameter_history[i + 1,j] = self.get_upper_off_diagonal_elements(self.correlation_matrix[j,:,:])
            self.initial_state_history[i + 1] = self.initial_states
            self.total_iterations += 1
            # Break
            if i > 1 and np.abs(self.log_likelihood_history[i] - self.log_likelihood_history[i-1]) < self.tolerance:
                print(f'Model Converged at iteration:  {i}')
                break
        pbar.close()


    def e_step(self):
        # Get the densities for the data 
        self.get_densities()

        # Forward Pass
        self.forward_pass()

        # Backward Pass
        self.backward_pass()

        # Smoothed Probabilities
        self.calculate_smoothed_probabilities()

        
        self.estimate_transition_matrix()


        



        
    def get_densities(self):
        # # Calculate Inverse and Determinant of the correlation matrix R
        # det_R = np.zeros(self.n_states)
        # inv_R = np.zeros((self.n_states,self.K,self.K))
        # for n in range(self.n_states):
        #     det_R[n] = np.linalg.det(self.correlation_matrix[n,:,:])
        #     inv_R[n,:,:] = np.linalg.inv(self.correlation_matrix[n,:,:])
        # # Use log for the determinant part to avoid overflow
        # log_determinants = np.sum(np.log(self.standard_deviations), axis=0)
        
        # # Initial log densities
        # initial_log_densities = np.ones((self.n_states, self.T)) * self.K * np.log(2 * np.pi)
        
        # # Combine initial log densities and log determinants
        # log_densities = initial_log_densities + log_determinants[np.newaxis, :]
        
        # # Adjust log_densities for broadcasting
        # det_R_adjusted = det_R[:, np.newaxis]
        
        # # Calculate z_t @ R_inv @ z_t' in a numerically stable manner
        # intermediate_result = np.einsum('nkl,lt->nkt', inv_R, self.residuals)
        # final_result = np.einsum('nkt,kt->nt', intermediate_result, self.residuals)
        
        # # Combine to update log densities
        # log_densities += -0.5 * (det_R_adjusted + final_result)
        
        # # Convert log densities to densities safely
        # max_log_densities = np.max(log_densities, axis=0, keepdims=True)
        # densities = np.exp(log_densities - max_log_densities)
 
        det_R = np.zeros(self.n_states)
        inv_R = np.zeros((self.n_states, self.K, self.K))
        densities = np.zeros((self.n_states, self.T))
        
        # Calculate determinant and inverse of R for each state
        for n in range(self.n_states):
            det_R[n] = np.linalg.det(self.correlation_matrix[n, :, :])
            inv_R[n, :, :] = np.linalg.inv(self.correlation_matrix[n, :, :])

        # Calculate the densities directly for each state and time point
        for n in range(self.n_states):
            for t in range(self.T):
                z_t = self.residuals[:, t]  # Residual at time t
                exponent_part = -0.5 * np.dot(z_t.T, np.dot(inv_R[n], z_t))
                # Calculate the density without log, note this could lead to underflow
                densities[n, t] = np.exp(exponent_part) / np.sqrt((2 * np.pi) ** self.K * det_R[n])
        
        self.densities = densities
        self.det_R = det_R
        self.inv_R = inv_R
        #self.densities = densities

    def forward_pass(self):# N, self., initial_states, densities, transition_matrix):
        # Initialize forward probabilities & Scale factors
        forward_probabilities = np.zeros((self.n_states, self.T))
        scale_factors = np.zeros(self.T)

        # Set observation 0
        forward_probabilities[:, 0] = self.initial_states * self.densities[:,0]
        scale_factors[0] = 1.0 / np.sum(forward_probabilities[:, 0], axis = 0)
        forward_probabilities[:,0] *= scale_factors[0, np.newaxis]

        # Loop through all self.T
        for t in range(1, self.T):
            forward_probabilities[:, t] = np.dot(forward_probabilities[:,t-1], self.transition_matrix) * self.densities[:,t]
            scale_factors[t] = 1.0 / np.sum(forward_probabilities[:, t], axis = 0)
            forward_probabilities[:,t] *= scale_factors[t, np.newaxis]
        # Return Scales and forward probabilities
        self.forward_probabilities = forward_probabilities
        self.scale_factors = scale_factors

    def backward_pass(self):
        # Initialize Backward probabilitiy array
        backward_probabilities = np.zeros((self.n_states, self.T))

        # set observation 0
        backward_probabilities[:, self.T-1] = 1.0 * self.scale_factors[self.T-1, np.newaxis]

        # Loop from self.T-2 to -1
        for t in range(self.T-2, -1, -1):
            backward_probabilities[:,t] = np.dot(self.transition_matrix, (self.densities[:,t+1] * backward_probabilities[:, t+1]))
            
            # Scale to prevent underflow
            backward_probabilities[:,t] *= self.scale_factors[t, np.newaxis]
        self.backward_probabilities = backward_probabilities


    def calculate_smoothed_probabilities(self):
        # Smoothed State probabilities
        numerator = self.forward_probabilities * self.backward_probabilities
        denominator = numerator.sum(axis=0, keepdims=True)
        u_hat = numerator / denominator

        # Initial state probabilities
        delta = u_hat[:,0]

        # Precompute smoothed transitions
        a = np.roll(self.forward_probabilities, shift=1, axis=1)
        a[:,0] = 0 # Set initial to 0 as there is no t-1 for the first element
        
        # Einsum over the precomputed
        numerator = np.einsum('jt,jk,kt,kt->jkt', a, self.transition_matrix, self.densities, self.backward_probabilities)
        denominator = numerator.sum(axis=(0,1), keepdims=True) + 1e-7 # Sum over both J and K for normalization
        v_hat = numerator / denominator

        # Return
        self.initial_states = delta
        self.u_hat = u_hat
        self.v_hat = v_hat



    def estimate_transition_matrix(self):
        f_ij = np.sum(self.v_hat, axis=2)
        f_ii = np.sum(f_ij, axis=0)
        # print(f'f_ij: \n{f_ij}')
        # print(f'f_ii: \n{f_ii}')
        transition_matrix = f_ij / f_ii
        # print(f'Transition: \n{transition_matrix}')
        # print(f'sum: \n{np.sum(transition_matrix, axis=0)}')
        self.transition_matrix = transition_matrix.T



    def m_step(self):
        current_likelihood = self.calculate_log_likelihood()
        matrix = self.estimate_model_parameters()
        correlation_matrix = np.zeros((self.n_states, self.K, self.K)) 
        D_matrix = np.zeros((self.n_states, self.K, self.K)) 
        for i in range(self.n_states):
            D_matrix = self.diagonal_matrix_operations(matrix[i,:,:])
            correlation_matrix[i,:,:] = self.cholesky_scale(D_matrix @ matrix[i,:,:] @ D_matrix)
        # print(f'Correlation Matrix: \n {correlation_matrix}')
        self.current_likelihood = current_likelihood
        self.correlation_matrix = correlation_matrix
        # print(self.correlation_matrix)

    def calculate_log_likelihood(self):
        # ll = np.log(np.sum(alpha[:,-1]))
        ll = np.sum(np.log(self.scale_factors)) #  np.sum(alpha[:,-1])
        return ll

    def estimate_model_parameters(self):
        sum_states = np.sum(self.u_hat, axis=-1)
        sum_states_reshaped = sum_states[:, np.newaxis, np.newaxis]
        correlation_matrix = np.sum(np.einsum('it,jt,nt->nijt', self.residuals, self.residuals, self.u_hat), axis=-1) / sum_states_reshaped
        # Regularization: Add a small value to the diagonal
        # epsilon = 0 # 1e-6  # Small positive value
        # for n in range(correlation_matrix.shape[0]):
        #     np.fill_diagonal(correlation_matrix[n], correlation_matrix[n].diagonal() + epsilon)
        
        return correlation_matrix

        # def density(residuals):
        #     K, T = residuals.shape
        #     correlation_matrix = np.sum(np.einsum('it,jt->ijt', residuals, residuals), axis=-1) / T
        #     return correlation_matrix
    def diagonal_matrix_operations(self, matrix):
        # Step 1: Extract the diagonal of the matrix
        diag_elements = np.diag(matrix)
        
        # Step 2: Create a new diagonal matrix from these elements
        diag_matrix = np.diag(diag_elements)
        
        # Step 3: Take the square root of the diagonal elements
        sqrt_diag_matrix = np.sqrt(diag_matrix)
        
        # Step 4: Compute the inverse of the matrix
        # Since it's a diagonal matrix, we can invert it by inverting each non-zero diagonal element
        inverse_matrix = np.linalg.inv(sqrt_diag_matrix)
        
        return inverse_matrix



    def num_corr(self):
        """
        Estimate the correlation matrices for each state using a Newton-type optimization.
        """
        # Flatten Parameters
        flattened_parameters = self.flatten_off_diagonals()
        # Define optimization result holder

        
        # Define bounds and constraints if necessary to ensure positive definiteness
        # Create a list of bounds for each parameter
        bounds = [(-0.99, 0.99) for _ in range(self.num_parameters * self.n_states)]

    
   
        # Optimize for each state's correlation matrix
        result = minimize(self.negative_log_likelihood, flattened_parameters,
                          method='L-BFGS-B',
                          bounds=bounds)  # Or another suitable method
                          # jac=True,  # If you provide the gradient
                          # hess=True
                          #options={'disp': True})  # If you provide the Hessian
        new_correlation_matrix = self.rebuild_matrices(result.x)
            # Reconstruct the correlation matrix from the optimization result
            # and ensure it's positive definite
        return new_correlation_matrix




    def numeric_corr(self):
        """
        Estimate the correlation matrices for each state using a Newton-type optimization.
        """
        # Flatten Parameters
        flattened_parameters = self.flatten_off_diagonals()
        # Define optimization result holder

        
        # Define bounds and constraints if necessary to ensure positive definiteness
        # Create a list of bounds for each parameter
        bounds = [(-0.99, 0.99) for _ in range(self.num_parameters * self.n_states)]

    
   
        # Optimize for each state's correlation matrix
        self.result = minimize(self.negative_log_likelihood, flattened_parameters,
                          method='L-BFGS-B',
                          bounds=bounds)  # Or another suitable method
                          # jac=True,  # If you provide the gradient
                          # hess=True
                          #options={'disp': True})  # If you provide the Hessian
        self.new_correlation_matrix = self.rebuild_matrices(self.result.x)
        self.numeric_log_likelihood = - self.result.fun
            # Reconstruct the correlation matrix from the optimization result
            # and ensure it's positive definite


    def flatten_off_diagonals(self):
        # Initialize an empty list to store off-diagonal values
        off_diagonals = []
        # Iterate over each state
        for i in range(self.n_states):
            matrix = self.correlation_matrix[i, :, :]
            # Extract upper triangular part without the diagonal (k=1)
            upper_tri = matrix[np.triu_indices(self.K, k=1)]
            off_diagonals.extend(upper_tri)
        # Convert list to a flat numpy array
        return np.array(off_diagonals)



    def negative_log_likelihood(self, flattened_parameters):
        """
        Calculate the negative log-likelihood for given correlation matrix parameters.
        """

        # Unpack arguments (e.g., data, model specifics)
        matrix = self.rebuild_matrices(flattened_parameters)

        correlation_matrix = np.zeros((self.n_states, self.K, self.K)) 
        D_matrix = np.zeros((self.n_states, self.K, self.K)) 
        for i in range(self.n_states):
            D_matrix = self.diagonal_matrix_operations(matrix[i,:,:])
            correlation_matrix[i,:,:] = D_matrix @ matrix[i,:,:] @ D_matrix 
            # correlation_matrix[i,:,:] = self.cholesky_scale(correlation_matrix[i,:,:])

        end_densities = self.get_final_densities(correlation_matrix)
        # The log likelihood calculation
        log_likelihood = np.sum(self.u_hat * end_densities)

        return -log_likelihood


 

    def rebuild_matrices(self, flat_array):
        # Initialize an array to store the rebuilt matrices
        rebuilt_matrices = np.zeros((self.n_states, self.K, self.K))
        # The number of off-diagonal elements in a symmetric matrix
        off_diagonal_count_per_state = int(self.K * (self.K - 1) / 2)
        # Iterate over each state
        for i in range(self.n_states):
            # Start and end indices in the flat array for the current state
            start_idx = i * off_diagonal_count_per_state
            end_idx = start_idx + off_diagonal_count_per_state
            # Extract the off-diagonal values for the current state
            off_diagonals = flat_array[start_idx:end_idx]
            # Fill the upper triangle of the matrix
            rebuilt_matrices[i][np.triu_indices(self.K, k=1)] = off_diagonals
            # Make the matrix symmetric by mirroring the upper triangle to the lower triangle
            rebuilt_matrices[i] = rebuilt_matrices[i] + rebuilt_matrices[i].T
            # Fill the diagonal with ones
            np.fill_diagonal(rebuilt_matrices[i], 1)
        return rebuilt_matrices




    # def get_final_densities(self, correlation_matrix):
 
    #     det_R = np.zeros(self.n_states)
    #     inv_R = np.zeros((self.n_states, self.K, self.K))
    #     densities = np.zeros((self.n_states, self.T))
        
    #     # Calculate determinant and inverse of R for each state
    #     for n in range(self.n_states):
    #         det_R[n] = np.linalg.det(correlation_matrix[n, :, :])
    #         inv_R[n, :, :] = np.linalg.inv(correlation_matrix[n, :, :])

    #     # Calculate the densities directly for each state and time point
    #     for n in range(self.n_states):
    #         for t in range(self.T):
    #             z_t = self.residuals[:, t]  # Residual at time t
    #             exponent_part = -0.5 * np.dot(z_t.T, np.dot(inv_R[n], z_t))
    #             # Calculate the density without log, note this could lead to underflow
    #             densities[n, t] = np.exp(exponent_part) / np.sqrt((2 * np.pi) ** self.K * det_R[n])
        
    #     return densities
    def get_final_densities(self, correlation_matrix):
        # Assuming correlation_matrix shape is (n_states, K, K)
        
        # Precompute log factors that do not depend on the state
        log_2piK = np.log(2 * np.pi) * self.K
        det_R_floor = 1e-10  # Floor value for determinants
        
        # Calculate determinants and inverses for all matrices
        det_R = np.linalg.det(correlation_matrix)
        # Use floor value for too small determinants to avoid negative logs
        log_det_R = np.log(np.maximum(det_R, det_R_floor))
        
        # Inverting all matrices
        # Note: There isn't a direct NumPy function for batch matrix inversion, so this is a compromise
        inv_R = np.linalg.inv(correlation_matrix)
        
        # Compute exponent parts
        # z_t.T @ inv_R[n] @ z_t for each t and n
        # This operation is inherently looped over T, as it involves dynamic slicing of self.residuals
        log_densities = np.empty((self.n_states, self.T))
        for t in range(self.T):
            z_t = self.residuals[:, t]  # Residual at time t across all K variables
            for n in range(self.n_states):
                exponent_part = -0.5 * np.dot(z_t.T, np.dot(inv_R[n], z_t))
                # Compute the log-density for each state and time point
                log_densities[n, t] = exponent_part - 0.5 * (log_2piK + log_det_R[n])

        # Check for inf in densities after exponentiation
        

        # densities = np.exp(log_densities)
       

        # if np.isinf(densities).any():
        #     print("Inf detected in densities")
        #     print("Detected at indices:", np.where(np.isinf(densities)))
        
        # if np.isinf(densities).any():
        #     # Replace inf values with a very large number, or handle as appropriate
        #     large_value = 1000#np.finfo(np.float64).max
        #     densities = np.where(np.isinf(densities), large_value, densities)
        #     # print("Inf values replaced in densities")

        return log_densities
  # def get_densities(self): 
  #       det_R = np.zeros(self.n_states)
  #       inv_R = np.zeros((self.n_states, self.K, self.K))
  #       densities = np.zeros((self.n_states, self.T))
        
  #       # Calculate determinant and inverse of R for each state
  #       for n in range(self.n_states):
  #           det_R[n] = np.linalg.det(self.correlation_matrix[n, :, :])
  #           inv_R[n, :, :] = np.linalg.inv(self.correlation_matrix[n, :, :])

  #       # Calculate the densities directly for each state and time point
  #       for n in range(self.n_states):
  #           for t in range(self.T):
  #               z_t = self.residuals[:, t]  # Residual at time t
  #               exponent_part = -0.5 * np.dot(z_t.T, np.dot(inv_R[n], z_t))
  #               # Calculate the density without log, note this could lead to underflow
  #               densities[n, t] = np.exp(exponent_part) / np.sqrt((2 * np.pi) ** self.K * det_R[n])
        
  #       self.densities = densities
  #       self.det_R = det_R
  #       self.inv_R = inv_R
        #self.densities = densities    # def get_final_densities(self, correlation_matrix):
    #     det_R = np.zeros(self.n_states)
    #     inv_R = np.zeros((self.n_states, self.K, self.K))
    #     log_densities = np.zeros((self.n_states, self.T))
        
    #     # Precompute log factors that do not depend on the state n
    #     log_2piK = np.log(2 * np.pi) * self.K
    #     det_R_floor = 1e-10
    #     for n in range(self.n_states):
    #         # Make sure the correlation matrix for state n is positive definite
    #         # This is crucial to avoid negative determinants
    #         # Adjust your correlation matrix here if necessary
            
    #         det_R[n] = np.linalg.det(correlation_matrix[n, :, :])
    #         if det_R[n] < det_R_floor:
    #             log_det_R = np.log(det_R_floor)  # Use floor value if determinant is too small
    #         else:
    #             log_det_R = np.log(det_R[n])
    #         inv_R[n, :, :] = np.linalg.inv(correlation_matrix[n, :, :])
            
    #         # Log of determinant should be handled carefully to avoid taking log of a negative number
    #         log_det_R = np.log(det_R[n]) if det_R[n] > 0 else -np.inf  # Fallback to a large negative number if det_R[n] is non-positive
            
    #         # Calculate the densities directly for each state and time point in log form
    #         for t in range(self.T):
    #             z_t = self.residuals[:, t]  # Residual at time t
    #             exponent_part = -0.5 * np.dot(z_t.T, np.dot(inv_R[n], z_t))
                
    #             # Compute the log-density
    #             log_density = exponent_part - 0.5 * (log_2piK + log_det_R)
    #             log_densities[n, t] = log_density
        
    #     # Returning exp of log densities to get back to densities
    #     # Be cautious with this step if your densities are very small as it might lead to underflow
    #     # It might be more numerically stable to work with log densities directly in subsequent calculations
    #     densities = np.exp(log_densities)
    #     if np.isinf(densities).any():
    #         print("Inf detected in densities")
    #         print("Current determinant:", det_R[n])
    #     print(densities)
    #     return densities
























    def plot_smoothed_probabilities(self, start_date=None, end_date=None):
        sns.set_palette('pastel')
        
        if start_date is not None and end_date is not None:
            date_range = pd.date_range(start=start_date, end=end_date, periods=self.T)
            x_values = date_range
            x_label = 'Time'
        else:
            x_values = range(self.T)
            x_label = 'Index'
        
        fig, axes = plt.subplots(self.n_states, 1, figsize=(14, 4 * self.n_states), sharex=True)
        
        if self.n_states == 1:
            axes = [axes]
        
        for i, ax in enumerate(axes):
            ax.fill_between(x_values, self.u_hat[i, :], label=f'State {i}', alpha=0.5)
            ax.set_title(f'Smoothed State Probabilities for State {i}', fontsize=10)  # Adjusted for subplot titles
            ax.set_xlabel(x_label)
            ax.set_ylabel('Probability')
            ax.legend(loc='upper right')
        
        if isinstance(x_values, pd.DatetimeIndex):
            fig.autofmt_xdate()
        
        fig.suptitle('Smoothed Probabilities', fontsize=24, y=0.95)  # Add main title at the top of the figure

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for the main title
        plt.savefig('Smoothed Probabilities.png')
        plt.show()


    def plot_combined_probabilities(self, start_date=None, end_date=None):
        sns.set_palette(['orange', 'lightblue', 'lightgreen'])  # Define a custom palette
        
        if start_date is not None and end_date is not None:
            date_range = pd.date_range(start=start_date, end=end_date, periods=self.T)
            x_values = date_range
            x_label = 'Time'
        else:
            x_values = range(self.T)
            x_label = 'Index'
        
        # Calculate the sum of probabilities across time for each state
        sums = self.u_hat.sum(axis=1)
        
        # Determine the order based on the sums
        order = np.argsort(sums)
        
        # Create the plot
        plt.figure(figsize=(14, 6))
        plt.title('Combined Smoothed State Probabilities',fontsize=24)
        
        # Plot the state with the largest sum (in the middle)
        middle_state = order[-2]
        plt.fill_between(x_values, 0, self.u_hat[middle_state, :], color='lightgreen', alpha=0.7, label=f'State {middle_state } (middle)')


        # Plot the state with the smallest sum (on the top)
        smallest_state = order[0]
        plt.fill_between(x_values, 1 - self.u_hat[smallest_state, :], 1, color='orange', alpha=0.5, label=f'State {smallest_state } (smallest)')

        # Plot the state with the second largest sum (at the bottom)
        largest_state = order[-1]
        plt.fill_between(x_values, self.u_hat[middle_state, :], self.u_hat[middle_state, :] + self.u_hat[largest_state, :], color='lightblue', alpha=0.7, label=f'State {largest_state } (largest)')
        
        plt.xlabel(x_label)
        plt.ylabel('Probability')
        plt.legend(loc='upper right')
        
        if isinstance(x_values, pd.DatetimeIndex):
            plt.gcf().autofmt_xdate()
        
        plt.tight_layout()
        plt.savefig('Combined Smoothed Probabilities.png')
        plt.show()

    # def plot_smoothed_probabilities(self):
    #     # Set the color palette to pastel
    #     sns.set_palette('pastel')
        
    #     # Create a figure and axis
    #     plt.figure(figsize=(14, 4))
        
    #     # Plot for each state
    #     for i in range(self.n_states):
    #         plt.figure(figsize=(14, 4))

    #         plt.fill_between(range(self.T), self.u_hat[i, :], label=f'State {i+1}', alpha=0.5)
    #         # plt.fill_between(range(self.T), self.u_hat[i, :], alpha=0.5)
    #         plt.title(f'Smoothed State Probabilities for State {i+1}')
    #         plt.xlabel('Time')
    #         plt.ylabel('Probability')
    #         plt.show()

    def plot_combined_probabilities(self, start_date=None, end_date=None):
        sns.set_palette(['orange', 'lightblue', 'lightgreen'])  # Define a custom palette
        
        if start_date is not None and end_date is not None:
            date_range = pd.date_range(start=start_date, end=end_date, periods=self.T)
            x_values = date_range
            x_label = 'Time'
        else:
            x_values = range(self.T)
            x_label = 'Index'
        
        # Calculate the sum of probabilities across time for each state
        sums = self.u_hat.sum(axis=1)
        
        # Determine the order based on the sums
        order = np.argsort(sums)
        
        # Create the plot
        plt.figure(figsize=(14, 6))
        plt.title('Combined Smoothed State Probabilities',fontsize=24)
        
        # Plot the state with the largest sum (in the middle)
        middle_state = order[-2]
        plt.fill_between(x_values, 0, self.u_hat[middle_state, :], color='lightgreen', alpha=0.7, label=f'State {middle_state } (middle)')


        # Plot the state with the smallest sum (on the top)
        smallest_state = order[0]
        plt.fill_between(x_values, 1 - self.u_hat[smallest_state, :], 1, color='orange', alpha=0.5, label=f'State {smallest_state } (smallest)')

        # Plot the state with the second largest sum (at the bottom)
        largest_state = order[-1]
        plt.fill_between(x_values, self.u_hat[middle_state, :], self.u_hat[middle_state, :] + self.u_hat[largest_state, :], color='lightblue', alpha=0.7, label=f'State {largest_state } (largest)')
        
        plt.xlabel(x_label)
        plt.ylabel('Probability')
        plt.legend(loc='upper right')
        
        if isinstance(x_values, pd.DatetimeIndex):
            plt.gcf().autofmt_xdate()
        
        plt.tight_layout()
        plt.savefig('Combined Smoothed Probabilities.png')
        plt.show()


    def plot_true_smoothed(self, extra_data):
        # Set seaborn style for better aesthetics
        sns.set(style='whitegrid')
        
        # Ensure extra_data's first column contains state information, and convert it to a numpy array
        states = extra_data.iloc[:, 0].unique()
        colors = sns.color_palette("pastel", len(states))
        
        # Create a color map based on states
        state_colors = {state: colors[i] for i, state in enumerate(states)}
        
        # Ensure u_hat is correctly oriented: states x time
        assert self.u_hat.shape[0] == len(states), "The first dimension of u_hat must match the number of states."
        
        # Create subplots for each state's smoothed probability
        fig, axes = plt.subplots(len(states), 1, figsize=(14, 4 * len(states)), sharex=True)
       
        
        # Make axes iterable if only one state/series
        if len(states) == 1:
            axes = [axes]
            
        for i, ax in enumerate(axes):
            # Plot the smoothed probabilities for state i
            ax.plot(self.u_hat[i, :], label=f'Smoothed Probabilities State {i}', linewidth=1.5)
            ax.set_ylabel(f'Smoothed Prob. State {i+1}')
            ax.legend(loc='upper right')
            
            # Shade the background based on true states from extra_data
            for t in range(extra_data.shape[0]):
                true_state = int(extra_data.iloc[t, 0])
                ax.axvspan(t, t+1, color=state_colors[true_state], alpha=0.3)  # Reduced alpha for visibility
        
        fig.xlabel('Time')
        fig.suptitle('Smoothed Probabilities', fontsize=24, y=0.95)  # Add main title at the top of the figure

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for the main title
        plt.tight_layout()
        plt.savefig('True Smoothed Probabilities.png')
        plt.show()




    def plot_histories(self):
        # Determine the total number of plots (2 for log likelihood and initial state probabilities + n_states for transition probabilities and parameter histories)
        total_plots = 2 + self.n_states * 2
        fig_height = 3 * total_plots  # Assuming each plot needs about 3 units of vertical space
        
        plt.figure(figsize=(12, fig_height))

        
        plot_index = 1  # To keep track of which subplot we're on
        
        # Plot transition probabilities for staying in each state
        for i in range(self.n_states):
            plt.subplot(total_plots, 1, plot_index)
            plt.plot(self.probability_history[:self.total_iterations + 1, i, i], label=f'Stay in State {i}')
            plt.title(f'Transition Probability from State {i} to State {i} ')
            plt.xlabel('Iteration')
            plt.ylabel('Probability')
            plt.legend(loc='upper right')
            plot_index += 1

        # Plot log likelihood history, skipping the first observation
        plt.subplot(total_plots, 1, plot_index)
        plt.plot(-self.log_likelihood_history[1:self.total_iterations + 1], label='Log Likelihood')
        plt.title('Log Likelihood History')
        plt.xlabel('Iteration')
        plt.ylabel('Log Likelihood')
        plt.legend(loc='upper right')
        plot_index += 1
        
        # Plot initial state probabilities history
        plt.subplot(total_plots, 1, plot_index)
        for i in range(self.n_states):
            plt.plot(self.initial_state_history[:self.total_iterations + 1, i], label=f'Initial State {i}')
        plt.title('Initial State Probabilities')
        plt.xlabel('Iteration')
        plt.ylabel('Probability')
        plt.legend(loc='upper right')
        plot_index += 1
        
        # Plot parameter history for each state
        for i in range(self.n_states):
            plt.subplot(total_plots, 1, plot_index)
            for j in range(self.num_parameters):
                plt.plot(self.parameter_history[:self.total_iterations + 1, i, j], label=f'Parameter {j}')
            plt.title(f'Parameters Estimation History: State {i}')
            plt.xlabel('Iteration')
            plt.ylabel('Parameter Value')
            plt.legend(loc='upper right')
            plot_index += 1
        plt.tight_layout()
        plt.suptitle('Smoothed Probabilities', fontsize=24, y=0.95)  # Add main title at the top of the figure

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for the main title
        plt.savefig('Parameter Histories.png')
        plt.show()

    # def plot_histories(self):
    #     # Set the plot size
    #     # Set the plot size
    #     plt.figure(figsize=(18, 6))
        
    #     # Plot transition probabilities
    #     for i in range(self.n_states):
    #         plt.subplot(1, self.n_states + 2, i + 1)
    #         for j in range(self.n_states):
    #             plt.plot(self.probability_history[:self.total_iterations + 1, i, j], label=f'Transition to {j+1}')
    #         plt.title(f'State {i+1} Transition Probabilities')
    #         plt.xlabel('Iteration')
    #         plt.ylabel('Probability')
    #         plt.legend(loc='best')
        
    #     # Plot log likelihood history
    #     plt.subplot(1, self.n_states + 2, self.n_states + 1)
    #     plt.plot(self.log_likelihood_history[:self.total_iterations + 1], label='Log Likelihood')
    #     plt.title('Log Likelihood History')
    #     plt.xlabel('Iteration')
    #     plt.ylabel('Log Likelihood')
    #     plt.legend(loc='best')
        
    #     # Plot initial state probabilities history
    #     plt.subplot(1, self.n_states + 2, self.n_states + 2)
    #     for i in range(self.n_states):
    #         plt.plot(self.initial_state_history[:self.total_iterations + 1, i], label=f'Initial State {i+1}')
    #     plt.title('Initial State Probabilities')
    #     plt.xlabel('Iteration')
    #     plt.ylabel('Probability')
    #     plt.legend(loc='best')
        
    #     # Adjust layout and show plot
    #     plt.tight_layout()
    #     plt.show()
        
    #     # Plot parameter history for each state
    #     for i in range(self.n_states):
    #         plt.figure(figsize=(12, 4))
    #         for j in range(self.num_parameters):
    #             plt.plot(self.parameter_history[:self.total_iterations + 1, i, j], label=f'Parameter {j+1}')
    #         plt.title(f'State {i+1} Parameters History')
    #         plt.xlabel('Iteration')
    #         plt.ylabel('Parameter Value')
    #         plt.legend(loc='best')
    #         plt.show()
    # def plot_correlation_matrices(self, both=False):
    #     n_rows = 2 if both else 1
    #     n_states = self.new_correlation_matrix.shape[0]

    #     fig, axes = plt.subplots(n_rows, n_states, figsize=(5 * n_states, 4 * n_rows))

    #     if n_rows == 1:
    #         axes = [axes]
        
    #     for i in range(n_states):
    #         ax = axes[0][i] if both else axes[i]
    #         sns.heatmap(self.new_correlation_matrix[i], ax=ax, annot=True, cmap='vlag_r', fmt=".2f",
    #                     xticklabels=self.labels, yticklabels=self.labels, vmin=-1, vmax=1)
    #         ax.set_title(f'Correlation Matrix State {i}')
    #         ax.xaxis.tick_top()
            
    #     if both:
    #         for i in range(n_states):
    #             sns.heatmap(self.correlation_matrix[i], ax=axes[1][i], annot=True, cmap='vlag_r', fmt=".2f",
    #                         xticklabels=self.labels, yticklabels=self.labels, vmin=-1, vmax=1)
    #             axes[1][i].set_title(f'Unoptimized Correlation Matrix State {i}')
    #             axes[1][i].xaxis.tick_top()
        
    #     fig.suptitle('Regime Switching Dynamic Correlation Matrices', fontsize=24, y=.95)
        
    #     # Adjust layout
    #     plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the rect parameter as needed to reduce the space
    #     plt.savefig('RSDC Heatmap.png')
    #     plt.show()






    # def plot_correlation_matrices(self, both=False):
    #     n_rows = 2 if both else 1
    #     n_states = self.new_correlation_matrix.shape[0]

    #     # Adjust figsize to help ensure square tiles, considering the number of rows and columns
    #     fig_width_per_state = max(len(self.labels), 5)  # Adjust based on the number of labels or desired width per heatmap
    #     fig_height_per_row = fig_width_per_state * 1 # Adjust the multiplier for height per row as needed
    #     fig, axes = plt.subplots(n_rows, n_states, figsize=(fig_width_per_state * n_states, fig_height_per_row * n_rows))

    #     if n_rows == 1:
    #         axes = np.array([axes])  # Ensure axes is always a 2D array for consistency
        
    #     for i in range(n_states):
   
    #         ax = axes[0][i] if both else axes[i]
    #         sns.heatmap(self.new_correlation_matrix[i], ax=ax, annot=True, cmap='vlag_r', fmt=".2f",
    #                     xticklabels=self.labels, yticklabels=self.labels, vmin=-1, vmax=1, cbar=False, square=True)
    #         ax.set_title(f'Correlation Matrix State {i}')
    #         ax.xaxis.tick_top()
            
    #     if both:
    #         for i in range(n_states):

    #             sns.heatmap(self.correlation_matrix[i], ax=axes[1][i], annot=True, cmap='vlag_r', fmt=".2f",
    #                         xticklabels=self.labels, yticklabels=self.labels, vmin=-1, vmax=1, cbar=False, square=True)
    #             axes[1][i].set_title(f'Unoptimized Correlation Matrix State {i}')
    #             axes[1][i].xaxis.tick_top()
        
    #     fig.suptitle('Regime Switching Dynamic Correlation Matrices', fontsize=24, y=0.95)
        
    #     # Adjust layout
    #     plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the rect parameter as needed to reduce the space
    #     plt.savefig('RSDC Heatmap.png')
    #     plt.show()

    def plot_correlation_matrices(self, both=False, gamma=1.0,column_spacing=0.5, y_spacing=1):
        n_rows = 2 if both else 1
        n_states = self.new_correlation_matrix.shape[0]

        # Scale figure dimensions by gamma
        base_width = 5  # Base width for each heatmap
        base_height = 4  # Base height per row of heatmaps
        fig_width = (base_width * n_states + 0.05) * gamma  # +0.05 for colorbar width, scaled by gamma
        fig_height = base_height * n_rows * gamma

        fig = plt.figure(figsize=(fig_width, fig_height))
        gs = gridspec.GridSpec(n_rows, n_states + 1, width_ratios=[1]*n_states + [0.05], wspace=column_spacing)  # Additional column for colorbar

        for row in range(n_rows):
            matrices_to_plot = self.new_correlation_matrix if row == 0 else self.correlation_matrix
            for i in range(n_states):
                ax = plt.subplot(gs[row, i])
                cbar_ax = plt.subplot(gs[row, -1]) if i == n_states-1 else None
                sns.heatmap(matrices_to_plot[i], ax=ax, annot=True, cmap='vlag_r', fmt=".2f",
                            xticklabels=self.labels, yticklabels=self.labels, vmin=-1, vmax=1, square=True,
                            cbar=i == n_states-1, cbar_ax=cbar_ax)
                ax.set_title(f'{"New " if row == 0 else "Unoptimized "}Correlation Matrix State {i}')
                ax.xaxis.tick_top()

        fig.suptitle('Regime Switching Dynamic Correlation Matrices', fontsize=24 * gamma, y=y_spacing)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig('RSDC Heatmap.png')
        plt.show()



    def compute_hessian_and_se(self):
        """
        Compute the Hessian and standard errors for the model parameters.
        
        Parameters:
        - model: The model object with a defined negative_log_likelihood method.
        
        Returns:
        - A tuple of (Hessian matrix, standard errors).
        """
        hessian_func = nd.Hessian(self.negative_log_likelihood)
        flattened_parameters = self.flatten_off_diagonals()
        hessian_estimated = hessian_func(flattened_parameters)
        covariance_matrix = np.linalg.inv(hessian_estimated)
        standard_errors = np.sqrt(np.abs(np.diag(covariance_matrix)))
        return (hessian_estimated, standard_errors)

    def optimization_results_to_latex_table_with_metrics(self, result, standard_errors, labels, n_observations, log_likelihood, aic, bic):
        """
        Generate a LaTeX table including model metrics and transition matrix.
        
        Parameters:
        - model: The model object containing the transition matrix.
        - result: The result object from scipy.optimize.minimize.
        - standard_errors: An array of standard errors.
        - labels: Parameter labels.
        - n_observations: Number of observations.
        - log_likelihood: Log Likelihood value.
        - aic: Akaike Information Criterion value.
        - bic: Bayesian Information Criterion value.
        
        Returns:
        - LaTeX table as a string.
        """
        latex_table = r"""\begin{{table}}[H]
    \centering
    \begin{{tabular}}{{|l r r|}}
    \hline
    \textbf{{Metric}} & \textbf{{Value}} & \\
    \hline
    Number of Observations & {n_observations} & \\
    Log Likelihood & {log_likelihood:.4f} & \\
    AIC & {aic:.4f} & \\
    BIC & {bic:.4f} & \\
    \hline
    \textbf{{Parameter}} & \textbf{{Value}} & \textbf{{Standard Error}} \\
    \hline
    """.format(n_observations=n_observations, log_likelihood=log_likelihood, aic=aic, bic=bic)

        # Parameter rows
        for idx, (label, value, error) in enumerate(zip(labels, result.x, standard_errors)):
            latex_table += f"{label} & {value:.4f} & {error:.4f} \\\\ \n"  # Note the double backslashes for new lines in LaTeX

        # Transition matrix rows
        latex_table += r"\hline" + "\n"
        latex_table += r"\textbf{{Transition Matrix}} & & \\" + "\n"
        latex_table += r"\hline" + "\n"
        for row in self.transition_matrix:
            latex_table += " & ".join(["State Transition"] + [f"{val:.4f}" for val in row]) + r" \\" + "\n"
        
        # Close the table
        latex_table += r"""\hline
    \end{{tabular}}
    \caption{{Optimization Results and Model Metrics}}
    \label{{tab:model_metrics}}
    \end{{table}}
    """
        return latex_table


    def print_results(self):    # Calculate the number of estimated parameters
        k = len(self.flatten_off_diagonals())

        # Compute Hessian and Standard Errors
        _, standard_errors = self.compute_hessian_and_se()
        
        # Calculate AIC and BIC (assuming self.log_likelihood is the maximized log-likelihood)
        aic = 2 * k - 2 * self.numeric_log_likelihood
        bic = np.log(self.T) * k - 2 * self.numeric_log_likelihood
        
        # Store AIC and BIC in the model object for access in the LaTeX table function
        self.aic = aic
        self.bic = bic

        # Generate LaTeX table with model metrics
        table = self.optimization_results_to_latex_table_with_metrics(
            result=self.result,
            standard_errors=standard_errors,
            labels=self.labels,  # Adjust as necessary
            n_observations=self.T,
            log_likelihood=self.numeric_log_likelihood,
            aic=self.aic,
            bic=self.bic
        )
        
        print(table)








































    def numeric_smoothed_probabilities(self, start_date=None, end_date=None):
        sns.set_palette('pastel')
        # Get the densities for the data 
        self.get_densities_1(self.new_correlation_matrix)
        self.numeric_transition_matrix = self.transition_matrix
        # Forward Pass
        self.forward_pass_1()

        # Backward Pass
        self.backward_pass_1()

        # Smoothed Probabilities
        self.calculate_smoothed_probabilities_1()

        self.estimate_transition_matrix_1()

        # # Forward Pass
        # self.forward_pass_1()

        # # Backward Pass
        # self.backward_pass_1()

        # # Smoothed Probabilities
        # self.calculate_smoothed_probabilities_1()
        

        if start_date is not None and end_date is not None:
            date_range = pd.date_range(start=start_date, end=end_date, periods=self.T)
            x_values = date_range
            x_label = 'Time'
        else:
            x_values = range(self.T)
            x_label = 'Index'
        
        fig, axes = plt.subplots(self.n_states, 1, figsize=(14, 4 * self.n_states), sharex=True)
        
        if self.n_states == 1:
            axes = [axes]
        
        for i, ax in enumerate(axes):
            ax.fill_between(x_values, self.u_hat[i, :], label=f'State {i}', alpha=0.5)
            ax.set_title(f'Smoothed State Probabilities for State {i}', fontsize=10)  # Adjusted for subplot titles
            ax.set_xlabel(x_label)
            ax.set_ylabel('Probability')
            ax.legend(loc='upper right')
        
        if isinstance(x_values, pd.DatetimeIndex):
            fig.autofmt_xdate()
        
        fig.suptitle('Smoothed Probabilities', fontsize=24, y=0.95)  # Add main title at the top of the figure

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for the main title
        plt.savefig('Smoothed Probabilities.png')
        plt.show()




    def get_densities_1(self, correlation_matrix):
        # Assuming correlation_matrix shape is (n_states, K, K)
        
        # Precompute log factors that do not depend on the state
        log_2piK = np.log(2 * np.pi) * self.K
        det_R_floor = 1e-10  # Floor value for determinants
        
        # Calculate determinants and inverses for all matrices
        det_R = np.linalg.det(correlation_matrix)
        # Use floor value for too small determinants to avoid negative logs
        det_R = np.maximum(det_R, det_R_floor)
        
        # Inverting all matrices
        # Note: There isn't a direct NumPy function for batch matrix inversion, so this is a compromise
        inv_R = np.linalg.inv(correlation_matrix)
        
        # Compute exponent parts
        # z_t.T @ inv_R[n] @ z_t for each t and n
        # This operation is inherently looped over T, as it involves dynamic slicing of self.residuals
        densities = np.empty((self.n_states, self.T))
        for t in range(self.T):
            z_t = self.residuals[:, t]  # Residual at time t across all K variables
            for n in range(self.n_states):
                exponent_part = np.exp(np.dot(z_t.T, np.dot(inv_R[n], z_t)))
                # Compute the log-density for each state and time point
                A = 1 / np.sqrt(exponent_part + 1e-7) 
                B = np.sqrt((2 * np.pi) ** self.K * det_R[n])
                densities[n, t] = A / B

        self.densities_1 = densities
        
        
    # def get_densities_1(self, matrix):
    #     det_R = np.zeros(self.n_states)
    #     inv_R = np.zeros((self.n_states, self.K, self.K))
    #     densities = np.zeros((self.n_states, self.T))
        
    #     # Calculate determinant and inverse of R for each state
    #     for n in range(self.n_states):
    #         det_R[n] = np.linalg.det(matrix[n, :, :])
    #         inv_R[n, :, :] = np.linalg.inv(matrix[n, :, :])

    #     # Calculate the densities directly for each state and time point
    #     for n in range(self.n_states):
    #         for t in range(self.T):
    #             z_t = self.residuals[:, t]  # Residual at time t
    #             exponent_part = -0.5 * np.dot(z_t.T, np.dot(inv_R[n], z_t))
    #             # Calculate the density without log, note this could lead to underflow
    #             densities[n, t] = np.exp(exponent_part) / np.sqrt((2 * np.pi) ** self.K * det_R[n])
        
    #     self.densities_1 = densities
    #     self.det_R_1 = det_R
    #     self.inv_R_1 = inv_R
    #     #self.densities = densities

    def forward_pass_1(self):# N, self., initial_states, densities, transition_matrix):
        # Initialize forward probabilities & Scale factors
        forward_probabilities = np.zeros((self.n_states, self.T))
        scale_factors = np.zeros(self.T)

        # Set observation 0
        forward_probabilities[:, 0] = self.initial_states * self.densities_1[:,0]
        scale_factors[0] = 1.0 / np.sum(forward_probabilities[:, 0], axis = 0)
        forward_probabilities[:,0] *= scale_factors[0, np.newaxis]

        # Loop through all self.T
        for t in range(1, self.T):
            forward_probabilities[:, t] = np.dot(forward_probabilities[:,t-1], self.numeric_transition_matrix) * self.densities_1[:,t]
            scale_factors[t] = 1.0 / np.sum(forward_probabilities[:, t], axis = 0)
            forward_probabilities[:,t] *= scale_factors[t, np.newaxis]
        # Return Scales and forward probabilities
        self.forward_probabilities = forward_probabilities
        self.scale_factors = scale_factors

    def backward_pass_1(self):
        # Initialize Backward probabilitiy array
        backward_probabilities = np.zeros((self.n_states, self.T))

        # set observation 0
        backward_probabilities[:, self.T-1] = 1.0 * self.scale_factors[self.T-1, np.newaxis]

        # Loop from self.T-2 to -1
        for t in range(self.T-2, -1, -1):
            backward_probabilities[:,t] = np.dot(self.numeric_transition_matrix, (self.densities_1[:,t+1] * backward_probabilities[:, t+1]))
            
            # Scale to prevent underflow
            backward_probabilities[:,t] *= self.scale_factors[t, np.newaxis]
        self.backward_probabilities = backward_probabilities


    def calculate_smoothed_probabilities_1(self):
        # Smoothed State probabilities
        numerator = self.forward_probabilities * self.backward_probabilities
        denominator = numerator.sum(axis=0, keepdims=True)
        u_hat = numerator / denominator

        # Initial state probabilities
        delta = u_hat[:,0]

        # Precompute smoothed transitions
        a = np.roll(self.forward_probabilities, shift=1, axis=1)
        a[:,0] = 0 # Set initial to 0 as there is no t-1 for the first element
        
        # Einsum over the precomputed
        numerator = np.einsum('jt,jk,kt,kt->jkt', a, self.numeric_transition_matrix, self.densities_1, self.backward_probabilities)
        denominator = numerator.sum(axis=(0,1), keepdims=True) + 1e-7 # Sum over both J and K for normalization
        v_hat = numerator / denominator

        # Return
        self.initial_states = delta
        self.u_hat = u_hat
        self.v_hat = v_hat



    def estimate_transition_matrix_1(self):
        f_ij = np.sum(self.v_hat, axis=2)
        f_ii = np.sum(f_ij, axis=0)
        # print(f'f_ij: \n{f_ij}')
        # print(f'f_ii: \n{f_ii}')
        transition_matrix = f_ij / f_ii
        # print(f'Transition: \n{transition_matrix}')
        # print(f'sum: \n{np.sum(transition_matrix, axis=0)}')
        self.numeric_transition_matrix = transition_matrix.T