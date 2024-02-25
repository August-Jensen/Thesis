import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import minimize
# =======================================================
# |                The Base Filter                      |
# =======================================================


class Base():
    """docstring for Base"""
    def __init__(self, dataframe, n_states=2):
        # Extract dataframe and column names to numpy array.
        self.data, self.labels = self.df_to_array(dataframe)
        self.n_states = n_states
        self.N, self.T = self.data.shape
        print(self.T)

    def df_to_array(self, dataframe):
        # Create Numpy Array
        data_array = dataframe.to_numpy().T

        # Get Column Labels
        labels = dataframe.columns.tolist()

        return data_array, labels

    # Find the log-likelihood contributions of the univariate volatility
    def univariate_log_likelihood_contribution(self, x, sigma):
        sigma = max(sigma, 1e-8)
        return -0.5 * np.log(2 * np.pi) - np.log(sigma) - (x ** 2) / (2 * sigma ** 2)

    def total_univariate_log_likelihood(self, GARCH_guess, x):
        self.x = x.T
        # Set Parameters
        omega, alpha, beta = GARCH_guess
        sigma = np.zeros(self.T)
        print(self.x.shape)
        print(sigma.shape)
        # Set the Initial Sigma to be Total Unconditional Variance of data
        sigma[0] = np.sqrt(np.var(x))
        print(sigma)

        # Calculate sigma[t] for the described model
        for t in range(1, self.T):
            sigma[t] = omega + alpha * np.abs(x[t-1]) + beta * np.abs(sigma[t-1])

        # Calculate the sum of the Log-Likelihood contributions
        univariate_log_likelihood = sum(self.univariate_log_likelihood_contribution(self.x[t], sigma[t]) for t in range(self.T))

        # Return the Negative Log-Likelihood
        return -univariate_log_likelihood


    def estimate_GARCH(self,x):
        # Initial Guess for omega, alpha, beta

        GARCH_guess = [0.002, 0.2, 0.7]
        def objective_function(GARCH_guess,):
            return self.total_univariate_log_likelihood(GARCH_guess)
        # Minimize the Negative Log-Likelihood Function
        result = minimize(fun=self.total_univariate_log_likelihood, x0=GARCH_guess, args=(self.x,), bounds=[(0, None), (0, 1), (0, 1)])
        #print(f"Estimated parameters: omega = {result.x[0]}, alpha = {result.x[1]}, beta = {result.x[2]}")

        # Set Parameters
        result_parameters = result.x

        # Return Parameters and Information
        return result_parameters, result

    def univariate_fit(self):
        univariate_estimates = []
        full_result = []

        for i in range(self.N):
            # Set initial guess for GARCH parameters
            self.x = self.data[:,i]



            # Estimate GARCH
            result, full = self.estimate_GARCH(self.x)
            
            # Append to list 
            univariate_estimates.append(result)
            full_result.append(full)

            # Print Results
            print(f"Time Series: {self.labels[i]}, \n    Estimated parameters: \n \t omega = {result[0]}, \n \t alpha = {result[1]}, \n \t beta = {result[2]}")

        # Create Arrays
        univariate_parameters = np.array(univariate_estimates)
        full_univariate = np.array(full_result)

        return univariate_parameters, full_univariate


















# =======================================================
# |                Initail and Basic                    |
# =======================================================























# =======================================================
# |                Initail and Basic                    |
# =======================================================























# =======================================================
# |                Initail and Basic                    |
# =======================================================























# =======================================================
# |                Initail and Basic                    |
# =======================================================













