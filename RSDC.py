import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize


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
