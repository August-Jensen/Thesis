import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

class Simulate:
    def __init__(self, num_series=1, num_obs=1000, n_states=2, deterministic=True, transition_diagonal=None,transition_matrix=None):
        """
        Should take num_series, num_obs, n_states, deterministic, 

        """
        # Setup Settings
        self.num_series = num_series
        self.num_obs = num_obs
        self.n_states = n_states

        # Manage Transition probabilities
        self.deterministic = deterministic
        self.transition_diagonal = transition_diagonal
        self.transition_matrix = transition_matrix
        self.transition_matrix = self.create_transition_matrix().T
        
        # Manage parameters

        # Tracking: We need to handle the following:
            # Data for current state, and each series in num_series.
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


    def setup_parameters(self,):
        pass 

    def Density(self,):
        pass

    def simulate(self):
        data = np.zeros((self.num_obs, self.num_series + 1))  # +1 for the state column
        current_state = np.random.choice(self.n_states)
        
        for t in range(self.num_obs):
            transition_probs = self.transition_matrix[:, current_state].T
            current_state = np.random.choice(self.n_states, p=transition_probs)
            data[t, 0] = current_state  # Store the current state
            
            for series in range(self.num_series):
                # Draw observation from normal distribution with mean 0 and sigma for the current state and series
                series_sigma = self.sigmas[series, current_state]
                observation = np.random.normal(0, series_sigma)
                data[t, series + 1] = observation
        
        self.full_data = pd.DataFrame(data, columns=['States'] + [f'Returns {i}' for i in range(self.num_series)])
        self.data = self.full_data.drop(columns=['States'])  # Remove 'States' column

    def plot_simulation(self, separate=True, cum=False):
        sns.set(style="whitegrid")
        plt.figure(figsize=(10, 6))

        # Choose a color palette
        palette = sns.color_palette("husl", self.num_series)
        state_colors = sns.color_palette("coolwarm", self.n_states)  # Adjusted for n_states

        if cum:
            data_to_plot = self.full_data.iloc[:, 1:].cumsum()  # Assuming first column is 'States'
        else:
            data_to_plot = self.full_data.iloc[:, 1:]  # Exclude 'States' column for plotting

        if separate:
            for i, series in enumerate(data_to_plot.columns):  # Adjusted to iterate over data_to_plot
                ax = plt.subplot(self.num_series, 1, i + 1)
                sns.lineplot(data=data_to_plot, x=data_to_plot.index, y=series, color=palette[i], ax=ax)
                plt.title(series)

                # Apply shading
                for state in range(self.n_states):
                    ax.fill_between(data_to_plot.index, data_to_plot[series].min(), data_to_plot[series].max(),
                                    where=self.full_data['States'] == state, color=state_colors[state], alpha=0.3)
        else:
            for i, series in enumerate(data_to_plot.columns):
                sns.lineplot(data=data_to_plot, x=data_to_plot.index, y=series, label=series, palette=palette)
            
            # Apply shading for combined plot
            for state in range(self.n_states):
                plt.fill_between(data_to_plot.index, data_to_plot.min().min(), data_to_plot.max().max(),
                                 where=self.full_data['States'] == state, color=state_colors[state], alpha=0.3)

        plt.tight_layout()
        plt.show()

class SV(Simulate):
    def __init__(self, sigmas=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigmas = sigmas
        self.setup_parameters()
        
    def setup_parameters(self):
        sigma_min = 0.1
        sigma_max = self.n_states * 2 - 1
        
        if self.sigmas is None:
            self.sigmas = np.zeros((self.num_series, self.n_states))
            for series in range(self.num_series):
                valid_sigmas = False
                while not valid_sigmas:
                    random_sigmas = np.random.uniform(sigma_min, sigma_max, self.n_states)
                    if self._has_minimum_spacing(random_sigmas, min_spacing=0.5):
                        np.random.shuffle(random_sigmas)  # Shuffle to ensure randomness in volatility order
                        self.sigmas[series, :] = random_sigmas
                        valid_sigmas = True
        else:
            self.sigmas = np.array(self.sigmas)
            assert self.sigmas.shape == (self.num_series, self.n_states), "Sigmas shape must match (num_series, n_states)"


    def _has_minimum_spacing(self, sigmas, min_spacing=0.5):



    def Density(self, t, series_sigma):
        if self.data is not None:
            y_t = self.data[t, 1:]  # Assuming first column is 'States', skip it
            log_likelihood_contribution = -0.5 * (np.log(2 * np.pi) + np.log(series_sigma**2) + (y_t**2) / (series_sigma**2))
            return np.exp(log_likelihood_contribution)
        else:
            return np.zeros(self.num_series) 







class AR(Simulate):
    def __init__(self, mu=None, phi=None, sigma=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mu = mu
        self.phi = phi
        self.sigmas = sigma
        self.setup_parameters()

    def setup_parameters(self):
        # Define ranges for parameters
        mu_range = (-0.1, 0.1)
        phi_range = (-1, 1)
        sigma_min = 0.1
        sigma_max = self.n_states * 2 - 1

        # Initialize or validate mu
        if self.mu is None:
            self.mu = np.random.uniform(mu_range[0], mu_range[1], self.n_states)
        else:
            self.mu = np.array(self.mu)
            assert len(self.mu) == self.n_states, "Mu must have a length equal to n_states"
        
        # Initialize or validate phi
        if self.phi is None:
            self.phi = np.random.uniform(phi_range[0], phi_range[1], self.n_states)
        else:
            self.phi = np.array(self.phi)
            assert len(self.phi) == self.n_states, "Phi must have a length equal to n_states"

        # Ensure mu[0] and mu[1], and phi[0] and phi[1] are 0.3 apart if there are exactly 2 states
        # if self.n_states == 2:
        #     while abs(self.mu[0] - self.mu[1]) < 0.05 or abs(self.phi[0] - self.phi[1]) < 0.3:
        #         self.mu = np.random.uniform(mu_range[0], mu_range[1], self.n_states)
        #         self.phi = np.random.uniform(phi_range[0], phi_range[1], self.n_states)

        # Initialize or validate sigma
        if self.sigmas is None:
            self.sigmas = np.zeros(self.n_states)
            for state in range(self.n_states):
                self.sigmas[state] = np.random.uniform(sigma_min, sigma_max)
        else:
            self.sigmas = np.array(self.sigmas)
            assert len(self.sigmas) == self.n_states, "Sigma must have a length equal to n_states"

    # Optionally, you might want to implement methods specific to AR models, such as simulation or density calculation methods, which would use mu, phi, and sigma.
    def Density(self, t):
        # Ensure there is a previous time step for t > 0
        if t == 0:
            raise ValueError("t must be greater than 0 to calculate the next step in the process.")
        
        # Assuming self.data holds the observed data and the first column is 'States'
        state = int(self.full_data.iloc[t-1, 0])  # Get the state at time t-1
        x_t_minus_1 = self.full_data.iloc[t-1, 1]  # Assuming the second column holds x[t-1] data
        
        # Calculate the error term from a normal distribution with mean 0 and variance sigma[state]
        error = np.random.normal(0, self.sigma[state])
        
        # Calculate the next step of the process
        x_t = self.mu[state] + self.phi[state] * x_t_minus_1 + error
        
        return x_t




    def simulate(self):
        data = np.zeros((self.num_obs, self.num_series + 1))  # +1 for the state column
        current_state = np.random.choice(self.n_states)
        
        # Initialize the first observation for each series, assuming x[0] = 0 for simplicity
        for series in range(self.num_series):
            data[0, series + 1] = 0  # Initial value of x[0] for each series
            
        for t in range(1, self.num_obs):  # Start from 1 since x[0] is already initialized
            transition_probs = self.transition_matrix[:, current_state].T
            current_state = np.random.choice(self.n_states, p=transition_probs)
            data[t, 0] = current_state  # Store the current state
            
            for series in range(self.num_series):
                # Use AR model dynamics to simulate the next observation
                x_t_minus_1 = data[t-1, series + 1]
                error = np.random.normal(0, self.sigmas[current_state])
                data[t, series + 1] = self.mu[current_state] + self.phi[current_state] * x_t_minus_1 + error
        
        self.full_data = pd.DataFrame(data, columns=['States'] + [f'Series {i}' for i in range(self.num_series)])
        self.data = self.full_data.drop(columns=['States'])  # Remove 'States' column

































# =======================================================
# |     Constant Conditional Correlation Estimator      |
# =======================================================
class CCCEstimator:
    def __init__(self, dataframe):
        self.data, self.labels = self.df_to_array(dataframe)
        self.N, self.T = self.data.shape
        print(self.data.shape)

    def df_to_array(self, dataframe):

        # Create Numpy Array
        data_array = dataframe.to_numpy().T
        

        # Get titles of columns for plotting
        labels = dataframe.columns.tolist()

        return data_array, labels


    # Find the log-likelihood contributions of the univariate volatility
    def univariate_log_likelihood_contribution(self, x, sigma):
        sigma = max(sigma, 1e-8)
        return -0.5 * np.log(2 * np.pi) - np.log(sigma) - (x ** 2) / (2 * sigma ** 2)


    # Calculate the total log-likelihood of the univariate volatility
    def total_univariate_log_likelihood(self, GARCH_guess, x):
        # Set Number of Observations

        # Set Parameters
        omega, alpha, beta = GARCH_guess
        sigma = np.zeros(self.T)

        # Set the Initial Sigma to be Total Unconditional Variance of data
        sigma[0] = np.sqrt(np.var(x))

        # Calculate sigma[t] for the described model
        for t in range(1, self.T):
            sigma[t] = omega + alpha * np.abs(x[t-1]) + beta * np.abs(sigma[t-1])

        # Calculate the sum of the Log-Likelihood contributions
        univariate_log_likelihood = sum(self.univariate_log_likelihood_contribution(x[t], sigma[t]) for t in range(T))

        # Return the Negative Log-Likelihood
        return -univariate_log_likelihood



    # Minimize - total log-likelihood of the univariate volatility
    def estimate_univariate_models(self, x):
        # Initial Guess for omega, alpha, beta
        GARCH_guess = [0.002, 0.2, 0.7]

        # Minimize the Negative Log-Likelihood Function
        result = minimize(fun=self.total_univariate_log_likelihood, x0=GARCH_guess, args=(x,), bounds=[(0, None), (0, 1), (0, 1)])
        #print(f"Estimated parameters: omega = {result.x[0]}, alpha = {result.x[1]}, beta = {result.x[2]}")

        # Set Parameters
        result_parameters = result.x

        # Set Variance-Covariance Hessian
        result_hessian = result.hess_inv.todense()  

        # Set Standard Errors
        result_se = np.sqrt(np.diagonal(result_hessian))


        # Return Parameters and Information
        return result_parameters, result_hessian, result_se

    # Get an array of univariate model parameters for all timeseries
    def estimate_univariate_parameters(self, data, labels):
        # Create list to store univariate parameters, hessians, and standard errors
        univariate_parameters = []
        # univariate_hessians = []
        # univariate_standard_errors = []

        # Iterate over each time series in 'data' and estimate parameters
        for i in range(self.N):  # data.shape[1] gives the number of time series (columns) in 'data'
            result_parameters, result_hessian, result_se = self.estimate_univariate_models(self.data[:, i])
            univariate_parameters.append(result_parameters)
            # univariate_hessians.append(result_hessian)
            # univariate_standard_errors.append(result_se)
            # Print the label and the estimated parameters for each time series
            print(f"Time Series: {labels[i]}, \n    Estimated parameters: \n \t omega = {result_parameters[0]}, \n \t alpha = {result_parameters[1]}, \n \t beta = {result_parameters[2]}")
        # Convert the lists of results to numpy arrayst 
        univariate_parameters_array = np.array(univariate_parameters)
        # univariate_hessians_array = np.array(univariate_hessians)
        # univariate_standard_errors_array = np.array(univariate_standard_errors)

        # Return the results
        return univariate_parameters_array# univariate_hessians_array, univariate_standard_errors_array

    def fit(self):
        params = self.estimate_univariate_parameters(self.data, self.labels)
        print(params)
        return params
#     # Forms the Correlation Matrix from RSDC_correlation_guess
#     def form_correlation_matrix(multi_guess):
#         # Determine the size of the matrix
#         n = int(np.sqrt(len(multi_guess) * 2)) + 1
#         if len(multi_guess) != n*(n-1)//2:
#             raise ValueError("Invalid number of parameters for any symmetric matrix.")
        
#         # Create an identity matrix of size n
#         matrix = np.eye(n)
        
#         # Fill in the off-diagonal elements
#         param_index = 0
#         for i in range(n):
#             for j in range(i + 1, n):
#                 matrix[i, j] = matrix[j, i] = multi_guess[param_index]
#                 param_index += 1
                
#         return matrix


#     # Calculate the Standard Deviations, sigma, from Univariate Estimates
#         # This could be done outside of the objective function? 
#     def calculate_standard_deviations(data, univariate_estimates):
#         # Get Data Dimensions
#         N,T = data.shape

#         # Create Array for Standard Deviations
#         standard_deviations = np.zeros((T,N))

#         # Calculate Sigmas for each timeseries
#         for i in range(N):
#             # Unpack Univariate Estimates
#             omega, alpha, beta = univariate_estimates[i]

#             # Create array for Sigma values
#             sigma = np.zeros(T)

#             # Set first observation of Sigma to Sample Variance
#             sigma[0] = np.sqrt(np.var(data[:, i]))

#             # Calculate Sigma[t]
#             for t in range(1, T):
#                 sigma[t] = omega + alpha * np.abs(data[i,t-1]) + beta * np.abs(sigma[t-1])

#             # Save Sigmas to Standard Deviation Array
#             standard_deviations[:, i] = sigma

#         # Return array of all Standard Deviations
#         return standard_deviations


#     # Creates a Diagonal Matrix of (N x N), with Standard Deviations on Diagonal, and zeros off the Diagonal
#     def create_diagonal_matrix(t, std_array):
#         """
#         Creates an N x N diagonal matrix with standard deviations at time t on the diagonal,
#         and zeros elsewhere. Here, N is the number of time series.

#         :param t: Integer, the time index for which the diagonal matrix is created.
#         :param standard_deviations: List of numpy arrays, each array contains the standard deviations over time for a variable.
#         :return: Numpy array, an N x N diagonal matrix with the standard deviations at time t on the diagonal.
#         """
#         # Extract the standard deviations at time t for each series
#         stds_at_t = np.array(std_array[t,:])
        
#         # Create a diagonal matrix with these values
#         diagonal_matrix = np.diag(stds_at_t)
        
#         return diagonal_matrix




#     # Check if a Correlation Matrix is PSD, Elements in [-1,1], and symmetric.
#     def check_correlation_matrix_is_valid(correlation_matrix):
#         # Check diagonal elements are all 1
#         if not np.all(np.diag(correlation_matrix) == 1):
#             return False, "Not all diagonal elements are 1."
        
#         # Check off-diagonal elements are between -1 and 1
#         if not np.all((correlation_matrix >= -1) & (correlation_matrix <= 1)):
#             return False, "Not all off-diagonal elements are between -1 and 1."
        
#         # Check if the matrix is positive semi-definite
#         # A matrix is positive semi-definite if all its eigenvalues are non-negative.
#         eigenvalues = np.linalg.eigvals(correlation_matrix)
#         if np.any(eigenvalues < -0.5):
#             print(eigenvalues)
#             return False, "The matrix is not positive semi-definite."
        
#         return True, "The matrix meets all criteria."
#     def ccc_likelihood_contribution(t, data, R, standard_deviations):
#         # What we need in the terms:
#         data = data.T
#         D = create_diagonal_matrix(t, standard_deviations)
#         # R is defined in Total CCC Likelihood 
        

#         # Linear Algebra
#         det_D = np.linalg.det(D)
#         inv_D = np.linalg.inv(D)
#         det_R = np.linalg.det(R)
#         inv_R = np.linalg.inv(R)

#         # The Shock Term
#         z = inv_D @ data[t]

#         # The Terms of the Log Likelihood Contribution
#         term_1 = N * np.log(2 * np.pi)
#         term_2 = 2 * np.log(det_D) 
#         term_3 = np.log(det_R)
#         term_4 = z.T @ inv_R @ z

#         log_likelihood_contribution = -0.5 * (term_1 + term_2 + term_3 + term_4)
#         return log_likelihood_contribution

#     def Hamilton_Filter(data,random_guesses, standard_deviations):
#         # Get Shape of Data
#         N, T = data.shape

#         # Form the Correlation Matrix
#         R = form_correlation_matrix(random_guesses)
#         # Array for Log-Likelihoods Contributions
#         log_likelihood_contributions = np.zeros(T)

#         # The For Loop
#         for t in range(T):
#             log_likelihood_contributions[t] = ccc_likelihood_contribution(t, data, R, standard_deviations)

#         negative_likelihood = - np.sum(log_likelihood_contributions)
#         #print(negative_likelihood)
#         # Return Negative Likelihood
#         return negative_likelihood   

#     def fit(data):
#         number_of_correlation_parameters = N * (N - 1) / 2
        
#         random_guesses = np.random.uniform(-0.5, 0.5, int(number_of_correlation_parameters)).tolist()
#         m_bounds = []
#         m_bounds += [(-0.99, 0.99)] * int(number_of_correlation_parameters)

#         print(random_guesses)
#         standard_deviations = np.zeros((N,T))
        
#         standard_deviations = calculate_standard_deviations(data, univ_params)
#         def objective_function(random_guesses):
#             return Hamilton_Filter(data,random_guesses, standard_deviations)
#         result = minimize(objective_function, random_guesses, bounds=m_bounds, method='L-BFGS-B')
#         return result

#     def plot_heatmaps(df, result_matrix, labels):
#         # Calculate the correlation matrix for the DataFrame
#         corr_matrix = df.corr()
#         dims, dimz = result_matrix.shape
#         print(dims)
#         # Set up the matplotlib figure with subplots
#         fig, ax = plt.subplots(1, 2, figsize=(18, 9))
        
#         # Plot the Unconditional Correlation heatmap
#         sns.heatmap(corr_matrix, ax=ax[0], annot=True, cmap='coolwarm')
#         ax[0].set_title('Unconditional Correlation')
        
#         # Plot the Conditional Correlation heatmap
#         sns.heatmap(result_matrix, ax=ax[1], annot=True, cmap='coolwarm', xticklabels=labels, yticklabels=labels)
#         ax[1].set_title('Conditional Correlation')
        
#         # Adjust layout for better appearance
#         plt.tight_layout()
        
#         # Save the figure
#         plt.savefig(f'f Heatmaps {dims}.png')
        
#         # Show the plot
#         plt.show()

#     # Example usage (note: you need to have a DataFrame `df` and a `result_matrix` variable ready for this to work):
#     # plot_side_by_side_heatmaps(df, result_matrix, labels)

#     # This function assumes you have a DataFrame `df`, a result matrix `result_matrix`, and a list of labels `labels`.
#     # Replace 'df', 'result_matrix', and 'labels' with your actual data variables when using this function.
#     plot_heatmaps(df, result_matrix, labels)



# # ===========================================================
# # |         Simulate the Data Generating Process            |
# # ===========================================================

# from abc import ABC, abstractmethod
# import numpy as np

# class BaseModel(ABC):
#     """
#     Abstract base class for simulation models.
#     """
#     @abstractmethod
#     def simulate(self):
#         pass

#     @abstractmethod
#     def update_parameters(self):
#         pass

# class MCMCSimulator:
#     """
#     Class for running MCMC simulations.
#     """
#     def __init__(self, model, n_states = 2, n_iterations = 1000, n_series = 2, transition_params = [0.95, 0.99]): # None):
#         self.model = model
#         self.n_states = n_states 
#         self.n_iterations = n_iterations
#         self.n_series = n_series
#         self.transition_params = transition_params


        
#         if len(transition_params) != n_states:
#             print(f'The number of transition parameters does not match the number of states. \n There are {len(transition_params)} transition parameters defined, but {n_states} states.')
        
#         else
#             self.transition_matrix = self.create_transition_matrix(transition_params)


#     def create_transition_matrix(self, transition_params):
#         """
#         Creates a transition matrix for a Markov chain.

#         Parameters:
#         - prob_stay: A list of probabilities of staying in the current state for each state.

#         Returns:
#         - A numpy array representing the transition matrix where each row sums to 1.
#         """
#         n_states = len(transition_params)
#         transition_matrix = np.zeros((n_states, n_states))
        
#         for i in range(n_states):
#             # Set the diagonal element
#             transition_matrix[i, i] = transition_params[i]
#             # Calculate the off-diagonal elements
#             off_diagonal_prob = (1 - transition_params[i]) / (n_states - 1) if n_states > 1 else 0
#             for j in range(n_states):
#                 if i != j:
#                     transition_matrix[i, j] = off_diagonal_prob
                    
#         return transition_matrix

#     def run_simulation(self, n_iterations):
#         for _ in range(n_iterations):
#             self.model.simulate()
#             self.model.update_parameters()

# class RSDCModel(BaseModel):
#     """
#     RSDC Model with state-dependent correlations.
#     """
#     def __init__(self, initial_state):
#         self.state = initial_state
#         # Initialize other parameters as needed

#     def simulate(self):
#         # Implement simulation logic
#         pass

#     def update_parameters(self):
#         # Implement parameter update logic
#         pass

# class GARCHModel(BaseModel):
#     """
#     Univariate GARCH Model.
#     """
#     def __init__(self, omega, alpha, beta):
#         self.omega = omega
#         self.alpha = alpha
#         self.beta = beta
#         # Initialize other GARCH parameters as needed

#     def simulate(self):
#         # Implement GARCH simulation logic
#         pass

#     def update_parameters(self):
#         # Implement parameter update logic specific to GARCH
#         pass





