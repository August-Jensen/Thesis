import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize



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





