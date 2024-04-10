# =================================================================================
# |      	For Estimating Hidden Markov Models & Markov Switching Models         |
# =================================================================================



import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

import yfinance as yf
from arch import arch_model
from tqdm import tqdm
from scipy.optimize import minimize
from scipy.stats import t, norm, skew, kurtosis
from scipy.stats.mstats import gmean


# ===========================================================
# |                     Base Model                          |
# ===========================================================

class Base:
    def __init__(self, dataframe, transition_guess=0.95, n_states=2, max_iterations=100, tolerance=1e-5):
        # Basic Model Settings
        self.max_iterations = max_iterations
        self.tolerance = tolerance       
        self.dataframe = dataframe

        # Model states
        self.n_states = n_states

        # Transform dataframe to numpy array & Get Labels
        self.data, self.labels = self.df_to_array(self.dataframe)

        # Data Dimensions 
        self.E, self.T = self.data.shape

        # Set Estimation Dimensions
            # 1 for a single output. Examples: univariate timeseries, RSDC, VAR,
            # self.K for multiple univariate timesereis under the same regime. 

        # Form Transition Matrix
        self.transition_matrix = self.create_transition_matrix(transition_guess)
        
        # Set Initial State Probability to 1/n_states
        self.initial_state_probabilities = np.ones(self.n_states) / self.n_states

        # Set parameters & settings specific to the model
        self.num_parameters = 1
        self.set_parameters()


        # Parameter, probability and Likelihood Histories
        self.parameter_history = np.zeros((self.max_iterations+1, self.num_parameters)) # Max_iterations + 1, number of parameters, n_states
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

    def df_to_array(self, dataframe):
        # Create Numpy Array
        data_array = dataframe.to_numpy().T

        # Get Column Labels
        labels = dataframe.columns.tolist()

        return data_array, labels

    def create_transition_matrix(self, diagonal):
        # Creates an N dimensional transition matrix. 
        # Create a matrix with diagonal values on the diagonal
        matrix_1 = diagonal * np.eye(self.n_states) 

        # Create a matrix with off diagonal values of 1-diagonal, and scale by n_states - 1
        matrix_2 = (1 - diagonal) * (np.ones((self.n_states, self.n_states)) - np.eye(self.n_states)) / (self.n_states - 1)
        transition_matrix = matrix_1 + matrix_2
        return transition_matrix

    # def set_parameters(self):
    #     overall_mean = np.mean(self.data)
    #     overall_std = np.std(self.data)

    #     params = np.zeros((self.n_states, 3)) # mu, phi (set to 0), sigma

    #     # Example strategy: distribute mu around the overall mean, and sigma as variations of overall std
    #     for state in range(self.n_states):
    #         params[state, 0] = overall_mean + np.random.uniform(-1, 1) * overall_std * 0.1 # mu
    #         params[state, 2] = overall_std * (1 + np.random.uniform(-0.1, 0.1)) # sigma

    #     self.parameters = params
    #     self.num_parameters = len(params.flatten())
    def set_parameters(self):
        # Setup Model Parameters

        # Initialize a 4D array: 3 (for mu, phi, sigma) x K x n_states
        self.parameters = np.zeros((self.n_states, 3))
        self.final_model_parameters = np.zeros((self.n_states, 3))
        # Set history for each iteration + 1, all 3 parameters
        self.parameter_history = np.zeros((self.max_iterations,self.n_states, 3))

        mean = np.mean(self.data[:])
        std = np.sqrt(np.var(self.data[:]))

        for state in range(self.n_states):
            # Perturb mean and standard deviation for diversity across states
            self.parameters[state, 0] = mean + (state - self.n_states / 2) * 0.01  # mu
            self.parameters[state, 2] = std + (state - self.n_states / 2) * 0.1  # sigma

        self.num_parameters = len(self.parameters.flatten())
        # for k in range(self.K):
        #     mean_k = np.mean(self.data[k, :])
        #     std_k = np.sqrt(np.var(self.data[k, :]))
            
        #     for state in range(self.n_states):
        #         # Perturb mean and standard deviation for diversity across states
        #         self.mu[k, state] = mean_k + (state - self.n_states / 2) * 0.01
        #         self.sigma[k, state] = std_k + (state - self.n_states / 2) * 0.1


    def fit(self):
        # Run the EM for loop
        for iteration in tqdm(range(self.max_iterations), desc='Fitting Model'):
            # Run the E-Step
            self.E_Step()

            # Run the M-Step
            self.M_Step()

            # Track Parameters
            self.Track(iteration)
            
            # Break if converged, using updated condition based on log likelihood change
            if iteration > 0 and np.abs(self.log_likelihood_history[iteration] - self.log_likelihood_history[iteration - 1]).max() < self.tolerance:
                print('The model converged!')
                break

    def E_Step(self):
        # Get Densities
        self.get_densities()

        # Forward Pass
        self.forward_pass()

        # Backward Pass
        self.backward_pass()

        # Calculate Smoothed State & Transition Probabilities
        self.calculate_smoothed_probabilities()

        # Calculate Initial State Probabilities
        # self.calculate_initial_state()

        # Calculate Transition Probabilities
        # self.calculate_transition_probabilities()

    def get_densities(self):
        # Initialize parameters
        mu = self.parameters[:, 0]
        phi = self.parameters[:, 1]
        sigma = self.parameters[:, 2]
        sigma = np.clip(sigma, 1e-6, np.inf)  # Ensure sigma > 0 to avoid division by zero
        
        # Reshape and prepare data
        X = self.data.reshape(self.E, self.T)  # Assuming this reshaping is intended as per original method
        X_lagged = np.roll(X, shift=1, axis=1)
        # Ensure the first element of X_lagged is handled correctly (e.g., set to 0 or some initial value)
        X_lagged[:, 0] = 0  # Assuming 0; adjust as needed

        # Calculate conditional means for each state using AR(1) model, across all times
        conditional_means = mu[:, np.newaxis] + phi[:, np.newaxis] * (X_lagged - mu[:, np.newaxis])

        # Calculate densities using the Gaussian PDF formula across all times
        # Note: Adjusting for independence across time; ensure this assumption fits your model
        densities = (1. / (sigma[:, np.newaxis] * np.sqrt(2 * np.pi))) * \
                    np.exp(-0.5 * ((X - conditional_means) / sigma[:, np.newaxis]) ** 2) + 1e-6

        # You might want to handle t=0 differently depending on your model specifics
        # For example, if t=0 should be ignored or treated with specific values, adjust accordingly
        # print(densities.shape)
        self.densities = densities  # Or assign to self.densities if that's required
    def Density(self, t, n):
        pass




    def M_Step(self):
        # Estimate parameters
        self.estimate_model_parameters()

        # Calculate the log likelihoods
        self.calculate_log_likelihood()

    def calculate_log_likelihood(self):
        """
        Calculate the log likelihood of the observed data given the model.
        
        Parameters:
        - scale_factors: Scaling factors from the forward pass, a 1D NumPy array of length T.
        
        Returns:
        - log_likelihood: The log likelihood of the observed data.
        """
        log_likelihood = np.sum(np.log(self.scale_factors))
        self.current_likelihood = log_likelihood   

    def forward_pass(self,):
        # Initialize the forward probability matrix forward_probabilities with shape (K, T, n_states)
        # and a scaling factor array with shape (K, T)
        forward_probabilities = np.zeros((self.n_states, self.T))
        scale_factors = np.zeros((self.T))
        
        # Initial step: Compute initial forward_probabilities values and scale them
        forward_probabilities[:, 0] = self.initial_state_probabilities * self.densities[:, 0]
        scale_factors[0] = 1.0 / np.sum(forward_probabilities[:, 0], axis=0)
        forward_probabilities[:, 0] *= scale_factors[0, np.newaxis]
        
        # Recursive step: Update forward_probabilities values for t > 0
        for t in range(1, self.T):
            # Vectorized update of forward_probabilities using matrix multiplication for transition probabilities
            forward_probabilities[:, t] = np.dot(forward_probabilities[:, t-1], self.transition_matrix) * self.densities[:, t]
            
            # Scale forward_probabilities values to prevent underflow
            scale_factors[t] = 1.0 / np.sum(forward_probabilities[:, t], axis=0)
            forward_probabilities[:, t] *= scale_factors[t, np.newaxis]
        
        self.forward_probabilities = forward_probabilities
        self.scale_factors = scale_factors


    def backward_pass(self,):
        # Initialize the backward probability matrix backward_probabilities with shape (K, T, n_states)
        backward_probabilities = np.zeros((self.n_states, self.T))
        
        # Initial step: Set the last backward_probabilities values to 1 and scale them by the last scale factors
        backward_probabilities[:, self.T-1] = 1.0 * self.scale_factors[self.T-1, np.newaxis]
        
        # Recursive step: Update backward_probabilities values from T-2 down to 0
        for t in range(self.T-2, -1, -1):
            backward_probabilities[:, t] = np.dot(self.transition_matrix, (self.densities[:, t+1] * backward_probabilities[:, t+1]))
            
            # Scale backward_probabilities values to prevent underflow, using the same scale factors as forward pass
            backward_probabilities[:, t] *= self.scale_factors[t, np.newaxis]
        
        self.backward_probabilities = backward_probabilities

    def calculate_smoothed_probabilities(self):
        # Smoothed state probabilities
        numerator = self.forward_probabilities * self.backward_probabilities  # Element-wise multiplication
        denominator = numerator.sum(axis=0, keepdims=True)  # Sum over states for normalization
        self.u_hat = numerator / denominator
        
        self.initial_state_probabilities = self.u_hat[:,0]
        # Precompute for smoothed transitions to avoid recomputation
        a = np.roll(self.forward_probabilities, shift=1, axis=1)  # Shift forward_probabilities by 1 to align t-1 with t
        a[:, 0] = 0  # Set initial shifted values to 0 as there's no t-1 for the first element
        G = self.transition_matrix
        d = self.densities
        b = self.backward_probabilities
        # Smoothed transition probabilities
        numerator = np.einsum('jt,jk,kt,kt->jkt', a, G, d,b)
        denominator = numerator.sum(axis=(0,1), keepdims=True) + 1e-6 # Sum over both j and k states for normalization
        self.v_hat = numerator / denominator


    def estimate_transition_matrix(self):
        f_ij = np.sum(self.v_hat, axis=2)
        f_ii = np.sum(f_ij, axis=0)
        r_ii = f_ii#.reshape((self.n_states, 1))
        # print(f'F_ij: {f_ij}\n shape{f_ij.shape}')
        # print(f'F_ii: {f_ii}\n shape{f_ii.shape}')
        T =f_ij/f_ii

        self.transition_matrix = T.T


    def estimate_model_parameters(self):
        params = np.zeros((self.n_states, 3))  # Initializing for mu, phi, sigma
        
        if self.data.ndim > 1:
            self.data = self.data.flatten()  # Ensure data is 1D
        
        for state in range(self.n_states):
            weights = self.u_hat[state, :]
            
            if weights.ndim > 1 or self.data.ndim > 1:
                print("Unexpected array dimensions:", weights.ndim, self.data.ndim)
                continue
            
            # Calculate weighted mean (mu)
            weighted_sum = np.dot(weights, self.data)
            total_weight = weights.sum()
            mu = weighted_sum / total_weight if total_weight > 0 else 0

            # Initialize sums for phi calculation
            phi_numerator, phi_denominator = 0.0, 0.0
            for t in range(1, self.T):  # Start from 1 since we use x[t-1]
                x_t = self.data[t] - mu
                x_t_minus_1 = self.data[t-1] - mu
                phi_numerator += weights[t] * x_t * x_t_minus_1
                phi_denominator += weights[t] * x_t_minus_1**2
            
            # Calculate phi
            phi = phi_numerator / phi_denominator if phi_denominator > 0 else 0

            # Weighted calculation for sigma (standard deviation)
            weighted_variance = np.dot(weights, (self.data - mu)**2) / total_weight if total_weight > 0 else 0
            sigma = np.sqrt(weighted_variance) if weighted_variance > 0 else 0

            # Assign calculated parameters for the current state
            params[state, :] = mu, phi, sigma

        self.parameters = params

    def Track(self, iteration):
        # Log-Likelihood History
        self.log_likelihood_history[iteration] = self.current_likelihood
        # np.zeros((self.max_iterations, self.K))
        self.initial_state_history[iteration,:] = self.initial_state_probabilities.copy()
        self.probability_history[iteration,:,:] = self.transition_matrix.copy()
        params = self.parameters.flatten()
        self.parameter_history[iteration,:] = params




# #  ===========================================================
# #  |           Regime Switching Dynamic Correlation          |
# #  ===========================================================

class RSDC(Base):
    def __init__(self, dataframe, univariate_parameters=None, *args, **kwargs):
        super().__init__(dataframe, *args, **kwargs)
        if univariate_parameters is None:
            self.univariate_parameters = np.zeros((self.E, 3))
            self.estimate_univariate_parameters()
        else:
            self.univariate_parameters = univariate_parameters

        self.num_parameters = int(self.E * (self.E - 1) / 2)

        # initialize parameters
        self.set_parameters()
        self.calculate_standard_deviations()
        self.calculate_standardized_residuals()
        self.form_correlation_matrix()

    def estimate_univariate_parameters(self):
        for e in range(self.E):
            # Extract the time series data for the current estimation
            ts_data = self.data[e, :].copy(order='C')
            
            # Fit the GARCH(1,1) model
            model = arch_model(ts_data, mean='Zero', vol='Garch', p=1, q=1)
            res = model.fit(update_freq=0, disp='off')
            
            # Store the estimated parameters: omega, alpha, beta
            omega = res.params['omega']
            alpha = res.params['alpha[1]']  # Accessing alpha parameter
            beta = res.params['beta[1]']  # Accessing beta parameter
            
            self.univariate_parameters[e, :] = [omega, alpha, beta]

    def set_parameters(self):
        # Assuming self.correlation_parameters is already defined correctly
        # Initialize rho as a numpy array with zeros for all states
        self.rho = np.zeros(self.num_parameters * self.n_states)  # Adjusted for a one-dimensional array
        
        # Set first half (corresponding to state 0) to 0.1
        first_half_end = self.num_parameters  # Calculate the end index for the first half
        self.rho[:first_half_end] = -0.3
        
        # Set second half (corresponding to state 1) to 0.0
        # This step might be redundant if the array is initialized with zeros, 
        # but it's included for clarity and in case the initialization value changes.
        second_half_start = first_half_end  # Calculate the start index for the second half
        self.rho[second_half_start:] = 0.3
        
        # Initialize parameter history
        self.parameter_history = np.zeros((self.max_iterations, self.num_parameters, self.n_states))

    def calculate_standard_deviations(self):
        # Preallocate sigma array with the shape of self.data
        sigmas = np.zeros_like(self.data)

        # Initial variance based on the historical data for each series
        initial_variances = np.var(self.data, axis=1)

        # Set initial variance for each series
        for e in range(self.E):
            sigmas[e, 0] = initial_variances[e]

        # Calculate sigmas for each time t using the appropriate model
        for t in range(1, self.T):
            for e in range(self.E):
                # GARCH
                sigmas[e, t] = self.univariate_parameters[e, 0] + self.univariate_parameters[e, 1] * self.data[e, t-1]**2 + self.univariate_parameters[e, 2] * sigmas[e, t-1]

        # If squared=False, take the square root for GARCH standard deviations
        sigmas = np.sqrt(sigmas)

        self.standard_deviations = sigmas


    def calculate_standardized_residuals(self):
        # The original method may have inaccuracies in inverting and multiplying matrices.
        # Correct approach for element-wise division to get standardized residuals:
        self.residuals = self.data / self.standard_deviations


    def form_correlation_matrix(self):
        # Assuming self.rho is a flat array containing all parameters for all states
        assert len(self.rho) == self.n_states * self.num_parameters, "Mismatch in the number of parameters and expected size"

        # Initialize arrays to store the results
        R_matrix = np.zeros((self.n_states, self.E, self.E))
        det_R = np.zeros(self.n_states)
        inv_R = np.zeros_like(R_matrix)

        # Process each state
        for state in range(self.n_states):
            # Calculate start and end index for the parameters of this state in self.rho
            start_idx = state * self.num_parameters
            end_idx = start_idx + self.num_parameters
            state_params = self.rho[start_idx:end_idx]

            # Fill the off-diagonal elements of the R matrix for this state
            R_matrix[state][np.diag_indices(self.E)] = 1  # Diagonal elements are 1
            triu_indices = np.triu_indices(self.E, 1)
            tril_indices = (triu_indices[1], triu_indices[0])  # Swap rows and cols for the lower triangle

            # Assuming state_params can fill the upper triangle off-diagonal
            R_matrix[state][triu_indices] = state_params
            R_matrix[state][tril_indices] = state_params  # Symmetric lower part

            # Compute determinant and inverse for this state's R matrix
            det_R[state] = np.linalg.det(R_matrix[state])
            inv_R[state] = np.linalg.inv(R_matrix[state])
 
        self.R_matrix = R_matrix
        self.det_R = det_R
        self.inv_R = inv_R
    def handle_overflow_warning(self, e, log_densities,densities, det_R, final_result):
        print("Custom Warning Handling: Overflow encountered in exp.")
        print("Max of densities:", np.max(densities))
        print("Max of log_densities:", np.max(log_densities))
        print("Max of det_R:", np.max(det_R))
        print("Max of final_result:", np.max(final_result))
        print("Min of densities:", np.min(densities))
        print("Min of log_densities:", np.min(log_densities))
        print("Min of det_R:", np.min(det_R))
        print("Min of final_result:", np.min(final_result))
        print("Custom Warning Handling: Overflow encountered in exp.")
        print("densities:", densities)
        print("det_R:", det_R)
        print("final_result:", final_result)

    def get_densities(self):
        # Create densities
        initial_densities = np.ones((self.n_states, self.T)) * self.E * np.log(2 * np.pi)
        # Create Determinants
        determinants = np.prod(self.standard_deviations, axis=0)
        
        # Combine initial densities and log of determinants
        densities = initial_densities + np.log(determinants)

        # Reshape det_R for broadcasting, with 
        det_R = self.det_R[:,np.newaxis]

        # Calculate z_t @ R_inv @ z_t'
        intermediate_result = np.einsum('nkl,lt->nkt', self.inv_R, self.residuals)
        final_result = np.einsum('nkt,kt->nt', intermediate_result, self.residuals)

        # Combine to create log_densities
        log_densities = -0.5 * (densities + det_R + final_result) 

        # Set self.densities
        self.densities = np.exp(log_densities)

        # Catch and handle warnings as errors within this context
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)  # Convert warnings to errors

            try:
                # Attempt to set self.densities, which may cause overflow
                self.densities = np.exp(log_densities)
            except RuntimeWarning as e:
                # Handle the specific overflow warning
                self.handle_overflow_warning(e, log_densities, densities, det_R, final_result)
                # You might want to handle the overflow condition, for example, by setting to a large number
                # or handling it according to your application's needs
            except Exception as e:
                # Handle other exceptions
                print(f"An unexpected error occurred: {e}")

    def scale_cholesky(self, matrix):
        # Decompose Matrix
        P = np.linalg.cholesky(matrix)

        # Use Sum squares method
        for j in range(self.E):
            sum_squares = np.sum(P[j, :j] ** 2)
            P[j, j] = np.sqrt(1 - sum_squares) if 1 - sum_squares > 0 else 0

        # Create the scaled matrix
        scaled = np.dot(P, P.T)

        return scaled

    def estimate_model_parameters(self):
        # Use Einsum to create (K x K) matrix of residuals, for all t. 
        # Then multiply by u_hat to scale by smoothed probability in each state.
        # This gives a (N x K x K x T) matrix 
        # Summing across T, To get (N x K x K)
        correlation_matrix = np.sum(np.einsum('it,jt,nt->nijt', self.residuals, self.residuals, self.u_hat), axis=-1)

        # Sum across u_hat to use in denominator
        sum_u_hat = np.sum(self.u_hat, axis=-1)

        # Reshape sum_u_hat for broadcasting
        reshaped_u = sum_u_hat[:, np.newaxis, np.newaxis]

        # Normalize each KxK vectpr by the sum of u_hat
        correlation_matrix = correlation_matrix / reshaped_u

        # Matrix to store Correct matrix
        np_corr = np.zeros((self.n_states,self.E,self.E))
        # Correct to ensure matrix is Positive Semi Definite
        for n in range(self.n_states):
            np_corr[n, :, :] = self.scale_cholesky(correlation_matrix[n,:,:])
        
        # Set self.R_matrix
        self.R_matrix = np_corr

        # Set self.parameters as the off diagonals
        self.parameters = self.handle_parameters(self.R_matrix)

        # Update Model parameters
        self.rho = self.parameters
        self.form_correlation_matrix()

    def handle_parameters(self, matrix):
        # Calculate indices for the upper triangle excluding the diagonal
        row_indices, col_indices = np.triu_indices(self.E, k=1)

        # Initialize an array to hold the flattened upper off-diagonal elements for each matrix
        off_diagonals_upper_flat = np.zeros((self.n_states, len(row_indices)))

        # Extract the upper off-diagonal elements for each matrix
        for n in range(self.n_states):
            off_diagonals_upper_flat[n] = matrix[n][row_indices, col_indices]

        # The result is a numpy array as requested
        return off_diagonals_upper_flat.flatten()


        # # Create an empty list to store the flat arrays of off-diagonal elements for each n
        # off_diagonals_flat = []

        # # Iterate over each n in n_states, and get off diagonal values
        # for n in range(N):
        #     # Create a mask for the off-diagonal elements
        #     mask = ~np.eye(K, dtype=bool)
        #     # Extract and flatten the off-diagonal elements using the mask
        #     off_diagonal = R_matrix[n][mask].flatten()
        #     # Store the flattened off-diagonal elements in the list
        #     off_diagonals_flat.append(off_diagonal)

        # # Convert the list to a numpy array for efficiency if needed
        # off_diagonals_flat_array = np.array(off_diagonals_flat)

        # off_diagonals_flat_array.shape, off_diagonals_flat_array

    def check_psd(self, matrix):
        # Check if the matrix is positive semi-definite
        eigenvalues = np.linalg.eigvalsh(matrix)  # Use eigvalsh since the matrix is symmetric
        is_psd = np.all(eigenvalues >= 0)

        return is_psd



    # def estimate_model_parameters(self):
    #     # Initialize R_matrix
    #     self.R_matrix = np.zeros((self.n_states, self.E, self.E))

    #     # Loop over states to calculate R_matrix for each state
    #     for n in range(self.n_states):
    #         numerator = np.zeros((self.E, self.E))
    #         denominator = 0
            
    #         # Loop over time points to accumulate numerator and denominator
    #         for t in range(self.T):
    #             Ut = self.residuals[:, t].reshape(-1, 1)  # Reshape U_t for outer product
    #             Ut_outer = Ut @ Ut.T  # Calculate outer product U_t U_t'
    #             weight = self.u_hat[n, t]  # Weight for the current state and time
                
    #             numerator += weight * Ut_outer  # Weighted sum of outer products
    #             denominator += weight  # Sum of weights (smoothed probabilities)
            
    #         # Avoid division by zero
    #         if denominator > 0:
    #             self.R_matrix[n] = numerator / denominator  # Normalize to get R_matrix for state n
    #         else:
    #             self.R_matrix[n] = np.eye(E)  # Fallback to identity matrix if denominator is 0

    #     # Additional code to handle optimization and parameter estimation as needed
    #     # This is a placeholder for where you'd include optimization code similar to the provided example