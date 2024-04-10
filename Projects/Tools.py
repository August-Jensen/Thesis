import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import matplotlib.colors as mcolors
import time
import warnings
# ========================================================
# |                 In This Document:                    |
# ========================================================

"""
Simulate is a Base class for simulation of data generating processes. 
    It allows for N states, and K different time series.
    It has A Costumizable Transition Matrix, and an Efficient Simulation Method.


"""

class SimBase:
    def __init__(self, K_series=1, num_obs=1000, n_states=2, deterministic=True, transition_diagonal=None,transition_matrix=None):
        """
        Should take K_series, num_obs, n_states, deterministic, 

        """
        # Setup Settings
        self.K_series = K_series
        self.num_obs = num_obs
        self.n_states = n_states

        # Manage Transition probabilities
        self.deterministic = deterministic
        self.transition_diagonal = transition_diagonal
        self.transition_matrix = transition_matrix
        self.transition_matrix = self.create_transition_matrix().T
        
        # Manage parameters

        # Tracking: We need to handle the following:
            # Data for current state, and each series in K_series.
            # A DataFrame with only the Observations.

    def create_transition_matrix(self):
        """
        Creates an (N x N) transition matrix. 4 Cases
            1. A Transition Matrix is already provided.
                    sets self.n_states to the length.
                    Returns the Transition Matrix
            2. The Diagonal Array of the Transition Matrix, or a Single Value is Provided.
                    If a Single Value is provided, it creates an array of self.n_states, with this value
                    If deterministic=True, it sets the off-diagonals to the same value, (1-diagonal) / (self.n_states-1)
                    If deterministic=False, it should draw the off diagonals at random
            3. If transition_diagonal=None, transition_matrix=None, and deterministic=True 
                    Create a transition matrix with 0.95 on diagonal, and off-diagonals all the same
            4. If transition_diagonal=None, transition_matrix=None, and deterministic=False
                    Draw each transition Probability at random, and 
        """
        # Case 0: If n_states = 1
        if self.n_states == 1:
            return np.ones(1)

        # Case 1: If a transition matrix is provided
        if self.transition_matrix is not None:
            self.n_states = len(self.transition_matrix)
            return np.array(self.transition_matrix)
        
        # Initialize an empty transition matrix
        transition_matrix = np.zeros((self.n_states, self.n_states))
        
        # Case 2: If transition_diagonal is provided
        if self.transition_diagonal is not None:
            if isinstance(self.transition_diagonal, (int, float)):
                diagonal_values = np.full(self.n_states, self.transition_diagonal)
            else:  # It's an array
                diagonal_values = np.array(self.transition_diagonal)
            
            np.fill_diagonal(transition_matrix, diagonal_values)
            
            for i in range(self.n_states):
                if self.deterministic:
                    off_diagonal_value = (1 - diagonal_values[i]) / (self.n_states - 1)
                    for j in range(self.n_states):
                        if i != j:
                            transition_matrix[i, j] = off_diagonal_value
                else:
                    row_sum = diagonal_values[i]
                    remaining_values = np.random.uniform(0, 1, self.n_states - 1)
                    remaining_values /= remaining_values.sum() / (1 - row_sum)
                    transition_matrix[i, np.arange(self.n_states) != i] = remaining_values
            
        # Case 3 and 4: If neither transition_matrix nor transition_diagonal is provided
        else:
            if self.deterministic:
                np.fill_diagonal(transition_matrix, 0.95)
                for i in range(self.n_states):
                    for j in range(self.n_states):
                        if i != j:
                            transition_matrix[i, j] = 0.05 / (self.n_states - 1)
            else:
                transition_matrix = np.random.uniform(0, 1, (self.n_states, self.n_states))
                transition_matrix /= transition_matrix.sum(axis=1)[:, np.newaxis]
        
        return transition_matrix







    def simulate(self):
        # Setup Data Array
        self.full_data = np.zeros((self.num_obs, self. K_series +1))

        # Determine the initial state.
        current_state = np.random.choice(self.n_states)

        # Run the Loop generating the data for each t:
        for t in range(self.num_obs):
            transition_probs = self.transition_matrix[:, current_state].T
            current_state = np.random.choice(self.n_states, p=transition_probs)
            # Set the state at time t
            self.full_data[t,0] = current_state

            # Run the loop for each series:
            for series in range(self.K_series):
                self.full_data[t, series+1] = self.Density(t, series)

        self.full_df = pd.DataFrame(self.full_data, columns=['States'] + [f'Returns {i}' for i in range(self.K_series)])
        self.data = self.full_df.drop(columns=['States'])  # Remove 'States' column




    def plot_simulation(self, separate=True, cum=False):    
        # Set seaborn style for better aesthetics
        sns.set(style='whitegrid')
        
        # Determine unique states for coloring
        states = np.unique(self.full_data[:, 0])
        colors = sns.color_palette("pastel", len(states))
        
        # Create a color map based on states
        state_colors = {state: colors[i] for i, state in enumerate(states)}
        
        # Plot each series in a separate subplot
        fig, axes = plt.subplots(self.K_series, 1, figsize=(14, 4 * self.K_series), sharex=True)
        
        if self.K_series == 1:
            axes = [axes]  # Make it iterable if only one series
        
        for i, ax in enumerate(axes):
            series_data = self.full_data[:, i + 1]
            ax.plot(series_data, label=f'Returns {i}', linewidth=1.5)
            ax.set_ylabel(f'Returns {i}')
            ax.legend(loc='upper right')
            
            # Shade the background based on states
            for t in range(self.num_obs):
                state = int(self.full_data[t, 0])
                ax.axvspan(t, t+1, color=state_colors[state], alpha=0.6)
        
        plt.xlabel('Time')
        plt.tight_layout()
        plt.show()



class SV(SimBase):
    def __init__(self, sigmas=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigmas=sigmas
        
        # Create Parameters
        self.setup_parameters()

    def setup_parameters(self):
        # Set Bounds on sigmas: max 3 for 2 states, 9 for 5 states. To differentiate
        sigma_min = 0.1
        sigma_max = self.n_states * 2 - 1

        if self.sigmas is None:
            self.sigmas = np.zeros((self.K_series, self.n_states))
            for series in range(self.K_series):
                valid_sigmas = False
                while not valid_sigmas:
                    random_sigmas = np.random.uniform(sigma_min, sigma_max, self.n_states)
                    if self._has_minimum_spacing(random_sigmas, min_spacing=0.5):
                        np.random.shuffle(random_sigmas)  # Shuffle to ensure randomness in volatility order
                        self.sigmas[series, :] = random_sigmas
                        valid_sigmas = True
        else:
            self.sigmas = np.array(self.sigmas)
            assert self.sigmas.shape == (self.K_series, self.n_states), "Sigmas shape must match (K_series, n_states)"

    def _has_minimum_spacing(self, sigmas, min_spacing=0.5):
        """Check if all differences between sorted sigmas are above min_spacing."""
        sorted_sigmas = np.sort(sigmas)
        return np.all(np.diff(sorted_sigmas) >= min_spacing)

    def Density(self, t, k):
        state = int(self.full_data[t, 0])
        sigma = self.sigmas[k, state]

        observation = np.random.normal(0,sigma)
        return observation


class AR(SimBase):
    """docstring for AR"""
    def __init__(self, mu=None, phi=None, sigmas=None, rw=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rw = rw
        self.mu = mu
        self.phi = phi
        self.sigmas = sigmas
        
        # Create Parameters
        self.setup_parameters()

    def setup_parameters(self):
        sigma_min = 0.1
        sigma_max = self.n_states * 2 - 1
        mu_range = (-0.1, 0.1)
        phi_range = (-0.7, 0.7) if not self.rw else (-1, 1)

        # Initialize mus
        self.mu = self._initialize_or_validate_parameter(self.mu, mu_range[0], mu_range[1], "mus", 0.05)
        
        # Initialize phis
        self.phi = self._initialize_or_validate_parameter(self.phi, phi_range[0], phi_range[1], "phis", 0.2)

        # Initialize or validate sigmas
        self.sigmas = self._initialize_or_validate_parameter(self.sigmas, sigma_min, sigma_max, "sigmas", 0.5)
        

    def _initialize_or_validate_parameter(self, parameter, min_val, max_val, parameter_name, min_spacing):
        if parameter is None:
            parameter = np.zeros((self.K_series, self.n_states))
            for series in range(self.K_series):
                valid_values = False
                while not valid_values:
                    random_values = np.random.uniform(min_val, max_val, self.n_states)
                    if self._has_minimum_spacing(random_values, min_spacing=min_spacing):
                        np.random.shuffle(random_values)  # Ensure randomness
                        parameter[series, :] = random_values
                        valid_values = True
        else:
            parameter = np.array(parameter)
            assert parameter.shape == (self.K_series, self.n_states), f"{parameter_name} shape must match (K_series, n_states)"
        return parameter

    def _has_minimum_spacing(self, values, min_spacing=0.5):
        sorted_values = np.sort(values)
        return np.all(np.diff(sorted_values) >= min_spacing)


    def Density(self, t, k):
        state = int(self.full_data[t, 0])
        mu = self.mu[k, state]
        phi = self.phi[k, state]
        sigma = self.sigmas[k, state]

        observation = mu + phi * self.full_data[t-1, k+1] + np.random.normal(0,sigma)
        return observation








class GARCH(SimBase):
    """docstring for AR"""
    def __init__(self, mu=None, phi=None, omega=None, alpha=None, beta=None, ar=False, rw=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ar = ar  # Flag to include AR terms or not
        self.rw = rw

        self.mu = mu
        self.phi = phi
        
        self.omega = omega
        self.alpha = alpha
        self.beta = beta

        self.sigmas = np.zeros((self.K_series, self.num_obs))  # Placeholder for volatility series

        # Create Parameters
        self.setup_parameters()

        if not self.ar:
            # Set mu and phi to 0 if AR terms are not included
            self.mu = np.zeros_like(self.mu)
            self.phi = np.zeros_like(self.phi)

        # Set initial Sigma.

    def setup_parameters(self):
        omega_range = (0, 0.3) if self.rw else (0, 0.1)
        alpha_range = (0, 1) if self.rw else (0.0, 0.3)  # Ensure alpha + beta < 1
        beta_range = (0, 1) if self.rw else (0.0, 0.7) # Ensure alpha + beta < 1

        mu_range = (-0.1, 0.1)
        phi_range = (-0.7, 0.7) if not self.rw else (-1, 1)

        # Initialize mus
        self.mu = self._initialize_or_validate_parameter(self.mu, mu_range[0], mu_range[1], "mus", 0.05)
        
        # Initialize phis
        self.phi = self._initialize_or_validate_parameter(self.phi, phi_range[0], phi_range[1], "phis", 0.1)

        # Initialize omega, alpha, beta
        self.omega = self._initialize_or_validate_parameter(self.omega, omega_range[0], omega_range[1], "omega", 0.02)
        self.alpha = self._initialize_or_validate_parameter(self.alpha, alpha_range[0], alpha_range[1], "alpha", 0.05)
        self.beta = self._initialize_or_validate_parameter(self.beta, beta_range[0], beta_range[1], "beta", 0.1)

    def _initialize_or_validate_parameter(self, parameter, min_val, max_val, parameter_name, min_spacing):
        if parameter is None:
            parameter = np.zeros((self.K_series, self.n_states))
            for series in range(self.K_series):
                valid_values = False
                while not valid_values:
                    random_values = np.random.uniform(min_val, max_val, self.n_states)
                    if self._has_minimum_spacing(random_values, min_spacing=min_spacing):
                        np.random.shuffle(random_values)  # Ensure randomness
                        parameter[series, :] = random_values
                        valid_values = True
        else:
            parameter = np.array(parameter)
            assert parameter.shape == (self.K_series, self.n_states), f"{parameter_name} shape must match (K_series, n_states)"
        return parameter

    def _has_minimum_spacing(self, values, min_spacing=0.5):
        sorted_values = np.sort(values)
        return np.all(np.diff(sorted_values) >= min_spacing)


    def Density(self, t, k):
        state = int(self.full_data[t, 0])
        mu = self.mu[k, state]
        phi = self.phi[k, state]
        omega = self.omega[k, state]
        alpha = self.alpha[k, state]
        beta = self.beta[k, state]
        
        self.sigmas[k, t] = np.sqrt(omega + alpha * (self.full_data[t-1, k+1] ** 2) + beta * (self.sigmas[k, t-1] ** 2))
        observation = mu + phi * self.full_data[t-1, k+1] + np.random.normal(0,1) * self.sigmas[k, t]
        return observation

class ARMACH(GARCH):
    """docstring for ARMACH"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.sigmas = np.zeros((self.K_series, self.num_obs))  # Placeholder for volatility series

        # Create Parameters
        self.setup_parameters()

        if not self.ar:
            # Set mu and phi to 0 if AR terms are not included
            self.mu = np.zeros_like(self.mu)
            self.phi = np.zeros_like(self.phi)
    
    def Density(self, t, k):
        state = int(self.full_data[t, 0])
        mu = self.mu[k, state]
        phi = self.phi[k, state]
        omega = self.omega[k, state]
        alpha = self.alpha[k, state]
        beta = self.beta[k, state]
        
        self.sigmas[k, t] = omega + alpha * np.abs(self.full_data[t-1, k+1]) + beta * np.abs(self.sigmas[k, t-1])
        observation = mu + phi * self.full_data[t-1, k+1] + np.random.normal(0,1) * self.sigmas[k, t]
        return observation




    
class RSDC(SimBase):
    def __init__(self, omega=None, alpha=None, beta=None, rho=None, square=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Drawing omega, alpha, and beta for each series
        self.omega = np.random.uniform(0, 0.2, self.K_series) if omega is None else omega
        self.alpha = np.random.uniform(0.05, 0.25, self.K_series) if alpha is None else alpha
        self.beta = np.random.uniform(0.74 - self.alpha, 1 - self.alpha, self.K_series) if beta is None else beta
        
        # Number of rho parameters needed
        self.rho_count = int(self.K_series * (self.K_series -1) / 2)

        # Determine whether to use GARCH or ARMACH. Sqare means Squared parameters.
        self.square = square

        # History of sigmas
        self.sigmas = np.zeros((self.K_series, self.num_obs))  # Placeholder for volatility series similar to GARCH model      
        self.rho_matrix = [self.initialize_correlation_matrix() for _ in range(self.n_states)]


        # Set the Density function based on the value of self.square
        self.set_density_function()


        # Initialize the array to hold the Cholesky decompositions with the correct shape
        # Assuming self.rho_matrix is a list of 2D numpy arrays, each being a square matrix
        if len(self.rho_matrix) > 0:
            matrix_shape = self.rho_matrix[0].shape
            self.cholesky_form = np.zeros((self.n_states, *matrix_shape))
        
            # Compute the Cholesky decomposition for each state's correlation matrix
            for i in range(self.n_states):
                self.cholesky_form[i] = np.linalg.cholesky(self.rho_matrix[i])


    def form_correlation_matrix(self, rho_values):
        """Forms a correlation matrix from a list of rho values."""
        assert len(rho_values) == self.rho_count, "Incorrect number of rho values"
        
        matrix = np.eye(self.K_series)  # Start with an identity matrix
        lower_tri_indices = np.tril_indices(self.K_series, -1)
        
        # Fill in the lower triangle
        matrix[lower_tri_indices] = rho_values
        
        # Mirror the lower triangle to the upper triangle
        matrix = matrix + matrix.T - np.eye(self.K_series)
        
        return matrix

    def is_positive_semi_definite(self, matrix):
        """Checks if a matrix is positive semi-definite."""
        return np.all(np.linalg.eigvals(matrix) >= 0)

    def initialize_correlation_matrix(self):
        """Generates and validates a PSD correlation matrix for each state."""
        while True:
            rho_values = np.random.uniform(-1, 1, self.rho_count)
            corr_matrix = self.form_correlation_matrix(rho_values)
            
            if self.is_positive_semi_definite(corr_matrix):
                return corr_matrix


    def plot_heatmaps(self):
        """Plots heatmaps of the correlation matrix for each state."""
        num_states = len(self.rho_matrix)
        fig, axes = plt.subplots(1, num_states, figsize=(num_states * 6, 5))

        # Adjust for the case when there is only one state to avoid indexing error
        if num_states == 1:
            axes = [axes]
        
        for i, matrix in enumerate(self.rho_matrix):
            ax = axes[i] if num_states > 1 else axes  # Select the appropriate axis for plotting
            sns.heatmap(matrix, ax=ax, cmap='Spectral', annot=True, fmt=".2f", 
                        cbar=i == 0,  # Only show color bar for the first plot to save space
                        square=True, vmin=-1, vmax=1)
            ax.set_title(f'State {i+1} Correlation Matrix')

        plt.tight_layout()
        plt.show()


    def set_density_function(self):
        """Sets the Density method to GARCH or ARMACH based on self.square."""
        if self.square:
            self.Density = self.GARCH
        else:
            self.Density = self.ARMACH

    def GARCH(self, t, k):
        """Density function for the GARCH model."""
        state = int(self.full_data[t, 0])
        omega = self.omega[k]
        alpha = self.alpha[k]
        beta = self.beta[k]
        
        self.sigmas[k, t] = np.sqrt(omega + alpha * (self.full_data[t-1, k+1] ** 2) + beta * (self.sigmas[k, t-1] ** 2))
        observation = np.random.normal(0, 1) * self.sigmas[k, t]
        return observation

    def ARMACH(self, t, k):
        """Density function for the ARMACH model."""
        state = int(self.full_data[t, 0])
        omega = self.omega[k]
        alpha = self.alpha[k]
        beta = self.beta[k]
        
        self.sigmas[k, t] = omega + alpha * np.abs(self.full_data[t-1, k+1]) + beta * np.abs(self.sigmas[k, t-1])
        observation = np.random.normal(0, 1) * self.sigmas[k, t]
        return observation


    def simulate(self):
        # Setup Data Array
        self.full_data = np.zeros((self.num_obs, self.K_series + 1))

        # Initialize an array to store the simulated returns for each series at each time step
        simulated_returns = np.zeros((self.num_obs, self.K_series))
        
        # Initialize an array to store uncorrelated simulated returns
        self.uncorrelated_data = np.zeros((self.num_obs, self.K_series + 1))

        # Determine the initial state
        current_state = np.random.choice(self.n_states)

        for t in range(self.num_obs):
            if self.n_states > 1:
                transition_probs = self.transition_matrix[:, current_state].T
                current_state = np.random.choice(self.n_states, p=transition_probs)
                self.full_data[t, 0] = current_state
                self.uncorrelated_data[t, 0] = current_state

            # Generate uncorrelated returns for each series
            for series in range(self.K_series):
                simulated_returns[t, series] = self.Density(t, series)

            # Save the uncorrelated returns before applying correlations
            self.uncorrelated_data[t, 1:] = simulated_returns[t, :].copy()

            # Apply the Cholesky matrix to the returns to introduce correlations
            simulated_returns[t, :] = np.dot(self.cholesky_form[current_state], simulated_returns[t, :])

        # Assign the simulated, now correlated, returns to the full_data array
        self.full_data[:, 1:] = simulated_returns
        # Create a DataFrame for easier handling and visualization
        self.full_df = pd.DataFrame(self.full_data, columns=['States'] + [f'Returns {i}' for i in range(self.K_series)])
        self.data = self.full_df.drop(columns=['States'])



    def plot_uncorrelated(self, separate=True, cum=False):    
        # Set seaborn style for better aesthetics
        sns.set(style='whitegrid')
        
        # Determine unique states for coloring
        states = np.unique(self.uncorrelated_data[:, 0])
        colors = sns.color_palette("pastel", len(states))
        
        # Create a color map based on states
        state_colors = {state: colors[i] for i, state in enumerate(states)}
        
        # Plot each series in a separate subplot
        fig, axes = plt.subplots(self.K_series, 1, figsize=(14, 4 * self.K_series), sharex=True)
        
        if self.K_series == 1:
            axes = [axes]  # Make it iterable if only one series
        
        for i, ax in enumerate(axes):
            series_data = self.uncorrelated_data[:, i + 1]
            ax.plot(series_data, label=f'Returns {i}', linewidth=1.5)
            ax.set_ylabel(f'Returns {i}')
            ax.legend(loc='upper right')
            
            # Shade the background based on states
            for t in range(self.num_obs):
                state = int(self.uncorrelated_data[t, 0])
                ax.axvspan(t, t+1, color=state_colors[state], alpha=0.6)
        
        plt.xlabel('Time')
        plt.tight_layout()
        plt.show()




class CCC:
    """docstring for CCC"""
    def __init__(self, dataframe, squared=False):
        self.dataframe = dataframe
        self.data, self.labels = self.df_to_array(self.dataframe)
        self.K, self.T = self.data.shape

        # Use Squared or Absolute term GARCH
        self.squared = squared
        self.set_density_function()

        # Initialize parameters array
        self.params = np.zeros((self.K, 3))  


    def df_to_array(self, dataframe):
        # Create Numpy Array
        data_array = dataframe.to_numpy().T

        # Get Column Labels
        labels = dataframe.columns.tolist()

        return data_array, labels

    def set_density_function(self):
        """Sets the Density method to GARCH or ARMACH based on self.squared."""
        if self.squared:
            self.Density = self.GARCH
        else:
            self.Density = self.ARMACH

    def garch_log_likelihood(self, params, data):
        omega, alpha, beta = params
        T = len(data)
        self.sigmas = np.zeros(T)
        self.sigmas[0] = np.var(data)

        for t in range(1, T):
            self.Density(t, params, data)  # Update self.sigmas[t] using the selected density function

        # Assuming data is the returns (x), compute the log likelihood
        log_likelihood = -np.sum(-np.log(self.sigmas) - data**2 / self.sigmas)
        return log_likelihood

    def GARCH(self, t, params, data):
        """Updates self.sigmas[t] based on the GARCH model."""
        omega, alpha, beta = params
        self.sigmas[t] = omega + alpha * data[t-1]**2 + beta * self.sigmas[t-1] + 1e-6

    def ARMACH(self, t, params, data):
        """Updates self.sigmas[t] based on the ARMACH model."""
        omega, alpha, beta = params
        self.sigmas[t] = omega + alpha * np.abs(data[t-1]) + beta * np.abs(self.sigmas[t-1]) + 1e-6

    def estimate_garch_parameters(self):
        self.results = {}
        for k, (label, series) in enumerate(zip(self.labels, self.data)):
            # Note: You may need to adjust bounds and initial guesses based on your data and model specifics
            res = minimize(self.garch_log_likelihood, x0=np.array([0.1, 0.1, 0.8]), args=(series,), method='L-BFGS-B', bounds=[(1e-8, None), (0, 1), (0, 1)])
            self.params[k, :] = res.x  # Store the optimized parameters in the params array
            self.results[label] = res
        
    def calculate_standard_deviations(self):
        # Preallocate sigma array with the shape of self.data
        sigmas = np.zeros_like(self.data)

        # Initial variance based on the historical data for each series
        initial_variances = np.var(self.data, axis=1)

        # Set initial variance for each series
        for k in range(self.K):
            sigmas[k, 0] = initial_variances[k]

        # Calculate sigmas for each time t using the appropriate model
        for t in range(1, self.T):
            for k in range(self.K):
                if self.squared:
                    # GARCH
                    sigmas[k, t] = self.params[k, 0] + self.params[k, 1] * self.data[k, t-1]**2 + self.params[k, 2] * sigmas[k, t-1]
                else:
                    # ARMACH
                    sigmas[k, t] = self.params[k, 0] + self.params[k, 1] * np.abs(self.data[k, t-1]) + self.params[k, 2] * np.abs(sigmas[k, t-1])

        # If squared=False, take the square root for GARCH standard deviations
        if self.squared:
            sigmas = np.sqrt(sigmas)

        self.standard_deviations = sigmas


    def calculate_standardized_residuals(self):
        # Ensure standard deviations are calculated
        if not hasattr(self, 'sigmas'):
            self.calculate_standard_deviations()

        # The original method may have inaccuracies in inverting and multiplying matrices.
        # Correct approach for element-wise division to get standardized residuals:
        self.residuals = self.data / self.standard_deviations


    def diagonalize_standard_deviations(self):
        D = np.zeros(self.T)
        for t in range(self.T):
            diagonalized = np.diag(self.standard_deviations[:,t])
            D[t] = np.linalg.det(diagonalized)
        # if np.min(D)<1e-6:
        #     print('Error, a D matrix is negative!')
        self.D_determinant = D

    def parameters_to_correlation_matrix(self, parameters):
        # Calculate the number of timeseries 'k' based on the length of parameters
        n = len(parameters)
        k = int(self.K)

        # Initialize the correlation matrix R with ones on the diagonal and zeros elsewhere
        R = np.eye(k)

        # Fill in the off-diagonal elements
        idx = np.triu_indices(k, 1)
        R[idx] = parameters
        R[(idx[1], idx[0])] = parameters  # Ensure symmetry

        return R


    def ccc_objective(self, parameters):
        log_likelihood = np.zeros(self.T)

        # Form the correlation matrix
        R_matrix = self.parameters_to_correlation_matrix(parameters)

        # Find the Determinant of the Correlation Matrix
        R_determinant = np.linalg.det(R_matrix) +1e-8

        # Find the Inverse of the Correlation Matrix
        R_inverse = np.linalg.inv(R_matrix)


        for t in range(self.T):
            term_1 = self.K * np.log(2 * np.pi)

            term_2 = 2 * np.log(self.D_determinant[t])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                # Your code that causes the warning goes here
                term_3 = np.log(R_determinant)


            term_4 = self.residuals[:,t].T @ R_inverse @ self.residuals[:,t]

            log_likelihood[t] = - 0.5 * (term_1 + term_2 + term_3 + term_4)



        nll = - np.sum(log_likelihood)

        return nll
    
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
    
    def minimize_correlation(self):

        # Calculate the number of parameters needed to form the correlation matrix R
        num_parameters = self.K * (self.K - 1) // 2

        # Define initial_parameters
        # For simplicity, starting with all parameters set to a small value close to 0,
        # indicating initial low correlation
        initial_parameters = np.zeros(num_parameters) + 0.01
        constraints = {'type': 'ineq', 'fun': self.optimization_constraint}
        bounds = [(-0.99, 0.99) for _ in range(num_parameters)]
        self.calculate_standard_deviations()
        self.calculate_standardized_residuals()
        self.diagonalize_standard_deviations()

        # During optimization, use a constraint
        def objective_function(parameters):
            return self.ccc_objective(parameters)


        self.ccc_estimate = minimize(
            objective_function,  # Your objective function
            initial_parameters,  # Initial guess of the parameters
            method='SLSQP',
            constraints=constraints,
            bounds=bounds
        )
        self.rho = self.ccc_estimate.x
        self.R_matrix = self.parameters_to_correlation_matrix(self.rho)


        

    def fit(self):
        self.estimate_garch_parameters()
