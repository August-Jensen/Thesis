import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import matplotlib.colors as mcolors
import time
# =====================================================================
# |         For Simulation Of Markov Chain Monte Carlo Models         |
# =====================================================================


class SimBase(object):
    """docstring for SimBase"""
    def __init__(self, K_series=1, num_obs=1000, n_states=2, deterministic=True, transition_diagonal=None, transition_matrix=None):
        # Setup Settings
        self.K_series = K_series
        self.num_obs = num_obs
        self.n_states = n_states

        # Manage Transition Probabilities
        self.deterministic = deterministic
        self.transition_diagonal = transition_diagonal
        self.transition_matrix = transition_matrix
        self.transition_matrix = self.create_transition_matrix().T


        self.stationary_distribution = self.calculate_stationary_distribution()


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
    #     # Case 0: If n_states = 1
    #     if self.n_states == 1:
    #         return np.ones(1)

    #     # Case 1: A Transition Matrix is Provided
    #     if self.transition_matrix is not None:
    #         self.n_states = len(self.transition_matrix)
    #         return np.array(self.transition_matrix)

    #     # Initialize an Empty Transition Matrix
    #     transition_matrix = np.zeros((self.n_states, self.n_states))

    #     # Case 2: If Transition Diagonal is Provided
    #     if self.transition_diagonal is not None:
    #         if isinstance(self.transition_diagonal, (int, float)):
    #             diagonal_values = np.full(self.n_states, self.transition_diagonal)
    #         else:
    #             diagonal_values = np.array(self.transition_diagonal)

    #             np.fill_diagonal(transition_matrix, diagonal_values)

    #             for i in range(self.n_states):
    #                 if self.deterministic:
    #                     off_diagonal_value = (1 - diagonal_values[i]) / (self.n_states - 1)
    #                     for j in range(self.n_states):
    #                         if i != j:
    #                             transition_matrix[i, j] = off_diagonal_value

    #                 else:
    #                     row_sum = diagonal_values[i]
    #                     remaining_values = np.random.uniform(0,1, self.n_states - 1)
    #                     remaining_values /= remaining_values.sum() / (1 - row_sum)
    #                     transition_matrix[i, np.arange(self.n_states) != i] =remaining_values

    #     # Case 3 & 4:  If Neither Transition Matrix nor Transition Diagonal is Provided
    #     else:
    #         if self.deterministic:
    #             np.fill_diagonal(transition_matrix, 0.95)
    #             for i in range(self.n_states):
    #                 for j in range(self.n_states):
    #                     if i != j:
    #                         transition_matrix[i,j] = 0.05 / (self.n_states - 1)
    #         else:
    #             transition_matrix = np.random.uniform(0, 1, (self.n_states, self.n_states))
    #             transition_matrix /= transition_matrix.sum(axis=1)[:, np.newaxis]
    #     print(self.transition_matrix)               
    #     return transition_matrix
    # def create_transition_matrix(self):
        # Case 0: If n_states = 1
        if self.n_states == 1:
            return np.ones((1, 1))

        # Case 1: A Transition Matrix is Provided
        if self.transition_matrix is not None:
            self.n_states = len(self.transition_matrix)
            return np.array(self.transition_matrix)

        # Initialize an Empty Transition Matrix
        transition_matrix = np.zeros((self.n_states, self.n_states))

        # Case 2: If Transition Diagonal is Provided
        if self.transition_diagonal is not None:
            if isinstance(self.transition_diagonal, (int, float)):
                # Single value provided, replicate it across the diagonal
                diagonal_values = np.full(self.n_states, self.transition_diagonal)
            else:
                # An array or list of values provided
                diagonal_values = np.array(self.transition_diagonal)
                if len(diagonal_values) != self.n_states:
                    raise ValueError("Length of transition_diagonal does not match n_states.")

            np.fill_diagonal(transition_matrix, diagonal_values)

            for i in range(self.n_states):
                if self.deterministic:
                    off_diagonal_value = (1 - diagonal_values[i]) / (self.n_states - 1)
                    transition_matrix[i, :] = off_diagonal_value
                    transition_matrix[i, i] = diagonal_values[i]  # Reinstate diagonal value after uniform distribution
                else:
                    # Ensure the sum of the row equals 1 by distributing the remainder randomly
                    remaining_indices = [j for j in range(self.n_states) if j != i]
                    remaining_values = np.random.uniform(0, 1, self.n_states - 1)
                    remaining_values /= remaining_values.sum()
                    remaining_values *= (1 - diagonal_values[i])
                    transition_matrix[i, remaining_indices] = remaining_values

        # Case 3 & 4: If Neither Transition Matrix nor Transition Diagonal is Provided
        else:
            if self.deterministic:
                np.fill_diagonal(transition_matrix, 0.95)
                for i in range(self.n_states):
                    for j in range(self.n_states):
                        if i != j:
                            transition_matrix[i, j] = 0.05 / (self.n_states - 1)
            else:
                # Assign random values and normalize
                transition_matrix = np.random.uniform(0, 1, (self.n_states, self.n_states))
                transition_matrix /= transition_matrix.sum(axis=1, keepdims=True)

        return transition_matrix.T


    def calculate_stationary_distribution(self):
        # should transpose if rows sum to 1.
        # A = np.transpose(self.transition_matrix) - np.eye(self.n_states)
        # Subtract identity matrix  
        A = self.transition_matrix - np.eye(self.n_states)

        # Add a row of ones for the condition that the sum is equal to 1.
        A = np.vstack((A, np.ones(self.n_states)))

        # We calculate the target vector as Ax = b with the last entry = 1
        # this satisfies sum of probabilities = 1 
        b = np.zeros(self.n_states + 1)
        b[-1] = 1

        # solve for the stationary distribution:
        stationary_distribution = np.linalg.lstsq(A, b, rcond=None)[0]

        return stationary_distribution

    def simulate(self):
        # Create Data Array
        self.full_data = np.zeros((self.num_obs, self.K_series + 1))

        # determine the initial state
        current_state = np.random.choice(self.n_states, p=self.stationary_distribution)

        # Run the Simulation Loop Generating the Data.
        for t in range(self.num_obs):
            if self.n_states > 1:
                transition_probs = self.transition_matrix[:, current_state].T
                current_state = np.random.choice(self.n_states, p=transition_probs)

                # Set the state at time t
                self.full_data[t,0] = current_state

            # Run the loop for each series
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



class SimSV(SimBase):
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


class SimAR(SimBase):
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








class SimGARCH(SimBase):
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

class SimARMACH(SimGARCH):
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




    
class SimRSDC(SimBase):
    def __init__(self, omega=None, alpha=None, beta=None, rho=None, square=True, *args, **kwargs):
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
        self.rho_matrix = [self.initialize_correlation_matrix() for _ in range(self.n_states)] if rho is None else rho

        self.correlation_matrix = np.array(self.rho_matrix)
        # Set the Density function based on the value of self.square
        self.set_density_function()
        self.T = self.num_obs

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




    # def vectorized_simulate_data(K, T, parameters, transition_matrix, cholesky):
    def my_cholesky_form(self, matrix):
        my_cholesky = np.zeros((self.n_states, self.K_series, self.K_series))
        for n in range(self.n_states):
            my_cholesky[n,:,:] = np.linalg.cholesky(matrix[n,:,:])

        return my_cholesky

    def simulate(self):
        # Pre-allocate arrays
        processes = np.zeros((self.K_series, self.T))
        variances = np.zeros((self.K_series, self.T))
        
        parameters = np.zeros((self.K_series, 3))
        parameters[:,0] = self.omega
        parameters[:,1] = self.alpha
        parameters[:,2] = self.beta        
        # print(f'Parameters: {parameters}')
        matrix = self.correlation_matrix
        my_cholesky = self.my_cholesky_form(matrix)
        states = np.zeros(self.T, dtype=int)
        innovations = np.random.normal(0, 1, (self.K_series, self.T))  # Pre-generate all innovations    
        # Initial variance
        variances[:, 0] = parameters[:, 0] / (1 - parameters[:, 1] - parameters[:, 2])
        
        # Simulate states with a Markov chain
        # This part is inherently sequential but let's try to minimize the loop's impact.
        choices = np.arange(self.n_states)
        for t in range(1, self.T):
            states[t] = np.random.choice(choices, p=self.transition_matrix[states[t-1]])
        
        # Vectorize the correlation application using pre-simulated innovations
        # Note: This approach changes the structure slightly, as we need to correlate all innovations first and then select based on state
        correlated_innovations = np.zeros((self.K_series, self.T))
        for state in range(self.transition_matrix.shape[0]):
            # Find indices where this state occurs
            state_indices = np.where(states == state)[0]
            if len(state_indices) > 0:
                for i in state_indices:
                    correlated_innovations[:, i] = my_cholesky[state] @ innovations[:, i]
        
        # Update variances and processes in a vectorized way
        for t in range(1, self.T):
            variances[:, t] = parameters[:, 0] + parameters[:, 1] * (processes[:, t-1]**2) + parameters[:, 2] * variances[:, t-1]
            processes[:, t] = np.sqrt(variances[:, t]) * correlated_innovations[:, t] 
        
        self.processes = processes.T
        self.states = states
        self.variances = variances
        self.innovations = innovations


        # Setup Data Array
        self.full_data = np.zeros((self.num_obs, self.K_series + 1))
        self.full_data[:, 0] = self.states
        self.full_data[:, 1:] = self.processes
        # # Initialize an array to store the simulated returns for each series at each time step
        # simulated_returns = np.zeros((self.num_obs, self.K_series))
        
        # # Initialize an array to store uncorrelated simulated returns
        # self.uncorrelated_data = np.zeros((self.num_obs, self.K_series + 1))

        # # Determine the initial state
        # current_state = np.random.choice(self.n_states)

        # for t in range(self.num_obs):
        #     if self.n_states > 1:
        #         transition_probs = self.transition_matrix[:, current_state].T
        #         current_state = np.random.choice(self.n_states, p=transition_probs)
        #         self.full_data[t, 0] = current_state
        #         self.uncorrelated_data[t, 0] = current_state

        #     # Generate uncorrelated returns for each series
        #     for series in range(self.K_series):
        #         simulated_returns[t, series] = self.Density(t, series)

        #     # Save the uncorrelated returns before applying correlations
        #     self.uncorrelated_data[t, 1:] = simulated_returns[t, :].copy()

        #     # Apply the Cholesky matrix to the returns to introduce correlations
        #     simulated_returns[t, :] = np.dot(self.cholesky_form[current_state], simulated_returns[t, :])

        # # Assign the simulated, now correlated, returns to the full_data array
        # self.full_data[:, 1:] = simulated_returns
        # # Create a DataFrame for easier handling and visualization
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



