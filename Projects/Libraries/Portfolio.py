import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class UtilityOptimize:
    """docstring for UtilityOptimize"""
    def __init__(self, initial_volatility, transition_matrix, univariate_parameters, correlation_matrix, mu, risk_free_return=0.001, initial_wealth=1, gamma=5, forecast_period=252, simulations=100):
        self.initial_volatility = initial_volatility 
        self.transition_matrix = transition_matrix 
        self.univariate_parameters = univariate_parameters 
        self.correlation_matrix = correlation_matrix 
        self.mu = mu 
        self.risk_free_return = risk_free_return
        self.initial_wealth = initial_wealth
        self.gamma = gamma
        self.forecast_period = forecast_period
        self.simulations = simulations

        # Set Shorthand
        self.N, self.K, self.E = correlation_matrix.shape 
        self.T = self.forecast_period
        self.S = self.simulations

        # Cholesky matrix
        self.cholesky_correlation = self.cholesky_form(self.correlation_matrix)




    def fit(self):
        '''
        Cholesky decomposition
        Gets portfolio returns 
        Runs minimize function
        '''
        self.get_portfolio_returns()

        self.result = self.minimize_utility()

    def cholesky_form(self, matrix):
        cholesky = np.zeros((self.N, self.K, self.K))
        for n in range(self.N):
            cholesky[n,:,:] = np.linalg.cholesky(matrix[n,:,:])
        return cholesky


    def get_portfolio_returns(self):
        """
        Simulates the required number of dataseries
        """
        mu_array = np.ones((self.K,self.T))
        for t in range(self.T):
            mu_array[:,t] = self.mu * mu_array[:,t] 

        # Initialize arrays to store simulations
        states = np.zeros((self.S, self.T), dtype=int)
        simulated_returns = np.zeros((self.S, self.K, self.T))
        simulated_volatility = np.zeros((self.S, self.K, self.T))

        # Simulate Data
        for s in range(self.S):
            processes, s_states, variances = self.simulate():

            #Store Results
            states[s,:] = s_states
            simulated_returns[s,:,:] = processes + mu_array
            simulated_volatility[s,:,:] = variances

        self.states = states
        self.simulated_returns = simulated_returns
        self.simulated_volatility = simulated_volatility

        self.cumulated_returns = np.cumsum(self.simulated_returns, axis=2)
    
    def simulate(self):
        """
        Simulates K timeseries for T periods, based on the GARCH(1,1) model, and state depemndent correlations
        

        Parameters:
        - K: Number of time series
        - T: Number of time periods
        - parameters: GARCH(1,1) parameters [omega, alpha, beta]
        - transition_matrix: Markov transition matrix for state changes
        - cholesky: Precomputed Cholesky matrices for each state
        - mu: Mean returns for each series
        - initial_vol: Initial variance for each series

        Returns:
        - processes: Simulated time series data
        - states: Simulated states for each time period
        - variances: Variance of each time series at each time period
        """
        # Pre-allocate Arrays
        processes = np.zeros((self.K, self.T))
        variances = np.zeros((self.K, self.T))
        states = np.zeros(self.T, dtype=int)
        innovations = np.random.normal(0, 1, (self.K, self.T))
        # Initial Variance
        variances[:,0] = self.initial_volatility
        # Simulate states with a Markov chain
        for t in range(1, self.T):
            states[t] = np.random.choice(np.arange(self.transition_matrix.shape[0]), p=self.transition_matrix[states[t-1]])

        # Apply Cholesky decomposition based on states and correlate innovations
        for state in np.unique(states):
            indices = np.where(states == state)[0]
            correlated_innovations = self.cholesky_correlation [state] @ innovations[:, indices]
            innovations[:, indices] = correlated_innovations
        
        # Update variances and processes in a vectorized way
        for t in range(1, self.T):
            variances[:, t] = self.univariate_parameters[:, 0] + self.univariate_parameters[:, 1] * (processes[:, t-1]**2) + self.univariate_parameters[:, 2] * variances[:, t-1]
            processes[:, t] = np.sqrt(variances[:, t]) * innovations[:, t]

        return processes, states, variances

    def minimize_utility(self, ):
        """
        Calculates the risky and risk free returns, T periods ahead.
        Then minimizes the objective function
        """
        self.cumulated_risk_free_return = np.exp(self.T * self.risk_free_return) 
        print(self.cumulated_returns[:,:, -1])
        
        self.exponential_cumulated_returns = np.exp(3)

        return result


    def objective(self):
        """
        Calculates weigted return
        Calculates the expected utility
        Calculates the mean expected utility
        return - mean returns
        """
        pass




def optimize_portfolio(initial_vol, transition_matrix, mean_returns, univariate_parameters, correlation_matrices, risk_free_return = 0.001, initial_wealth=1, gamma=5, forecast_period=252, simulations=1000):
    # Get The Data for optimization
    # mean_returns, mean_volatility, mean_states, mean_cum = get_portfolio_returns(initial_vol,transition_matrix, mean_returns, univar_params, correlaitons, simulations=1000, forecast_period=252)
    N, K, E = correlation_matrices.shape
    # Maximize ut    T = forecast_periodility by integration over sample paths
    weights = np.random.uniform(low=0, high=1, size=K) / K
    identity_vector = np.ones(K)
    risk_free_weight = 1 - weights.dot(identity_vector)
    T = forecast_period
    weighted_risk_free_return = risk_free_weight * np.exp(T * risk_free_return)
    # expected_wealth = 
    print(weighted_risk_free_return)
    
    bounds = [(0, 1)] * K
    # mean_returns, mean_volatility, mean_states, mean_cum = get_portfolio_returns(initial_vol,transition_matrix, mean_returns, univar_params, correlaitons, simulations=1000, forecast_period=252)
    return mean_returns, mean_volatility, mean_states, mean_cum

def minimize_utility(wealth=1, gamma=5):
    utility = (wealth ** (1-gamma)) / (1 - gamma) 
    
    return utility


# # =======================================================================
# # |                   Portfolio Simulation Model                       |
# # =======================================================================



    
    
# def simulate(K, T, parameters, transition_matrix, cholesky, mu, initial_vol):
#     # Pre-allocate arrays
#     processes = np.zeros((K, T))
#     variances = np.zeros((K, T))
#     states = np.zeros(T, dtype=int)
#     innovations = np.random.normal(0, 1, (K, T))  # Pre-generate all innovations
    
    
#     print(mu.shape)
#     mu = mu.reshape(-1, 1)
#     print(mu.shape)
#     # Initial variance
#     variances[:, 0] = initial_vol
#     print(variances[:,0])
#     # Simulate states with a Markov chain
#     # This part is inherently sequential but let's try to minimize the loop's impact.
#     choices = np.arange(transition_matrix.shape[0])
#     for t in range(1, T):
#         states[t] = np.random.choice(choices, p=transition_matrix[states[t-1]])
    
#     # Vectorize the correlation application using pre-simulated innovations
#     # Note: This approach changes the structure slightly, as we need to correlate all innovations first and then select based on state
#     correlated_innovations = np.zeros((K, T))
#     for state in range(transition_matrix.shape[0]):
#         # Find indices where this state occurs
#         state_indices = np.where(states == state)[0]
#         if len(state_indices) > 0:
#             for i in state_indices:
#                 correlated_innovations[:, i] = cholesky[state] @ innovations[:, i]
    
#     # Update variances and processes in a vectorized way
#     for t in range(1, T):
#         variances[:, t] = parameters[:, 0] + parameters[:, 1] * (processes[:, t-1]**2) + parameters[:, 2] * variances[:, t-1]
#         processes[:, t] = np.sqrt(variances[:, t]) * correlated_innovations[:, t]
    
#     return processes, states, variances#, innovations


# p = max_utility()



# initial_vol = model.standard_deviations[:, -1]
# transition_matrix = model.transition_matrix

# univar_params = model.univariate_parameters
# correlaitons = model.correlation_matrix

# # 
# mean_returns, mean_volatility, mean_states, mean_cum = optimize_portfolio(initial_vol,transition_matrix, mean_returns, univar_params, correlaitons, simulations=1000, forecast_period=252)













# # =======================================================================
# # |      	For Protfolio Optimization, VaR, Forecasting & Risk         |
# # =======================================================================

# # Optimize Portfolio Weights
# import numpy as np
# import matplotlib.pyplot as plt

# # Define the expected returns, standard deviations, and correlation matrix for the assets
# expected_returns = np.array([0.10, 0.12, 0.14])  # Expected returns for Asset 1, Asset 2, Asset 3
# std_devs = np.array([0.15, 0.20, 0.25])  # Standard deviations for Asset 1, Asset 2, Asset 3

# # Assume some correlation coefficients for simplicity
# correlation_matrix = np.array([[1, 0.8, 0.65],
#                                [0.8, 1, 0.75],
#                                [0.65, 0.75, 1]])

# # Calculate the covariance matrix from the standard deviations and correlation matrix
# cov_matrix = np.outer(std_devs, std_devs) * correlation_matrix

# # Generate random portfolio weights
# num_portfolios = 10000
# weights = np.random.random(size=(num_portfolios, 3))
# weights /= np.sum(weights, axis=1)[:, np.newaxis]

# # Correcting the calculation for portfolio volatility
# portfolio_variances = np.array([weights[i].dot(cov_matrix).dot(weights[i].T) for i in range(num_portfolios)])
# portfolio_volatility = np.sqrt(portfolio_variances)

# # Plotting the efficient frontier with the corrected volatility calculation
# plt.scatter(portfolio_volatility, portfolio_returns, c=(portfolio_returns-0.02)/portfolio_volatility, marker='o')
# plt.grid(True)
# plt.xlabel('Expected Volatility')
# plt.ylabel('Expected Return')
# plt.colorbar(label='Sharpe Ratio')
# plt.title('Efficient Frontier for 3 Assets')
# plt.show()

# # Include model for volatility, Correlations etc. (GARCH)

# # Efficient frontier

# # Forecast

# # Variance At Risk

# # Expected Shortfall
def optimize_portfolio(transition_matrix, mean_return, univariate_parameters, correlation_matrices, initial_wealth=1, gamma=5, forecast_period=252, simulations=1000):
    # Forecast asset returns
    # Integrate over
    N, K, E = correlation_matrices.shape
    T = forecast_period
    S = simulation

    # Setup the innovations, Dimensions (N x K x T x S)
    innovations = np.random.randn(N, K, T, S)

    # Simulate a GARCH process
    returns, volatility = simulate_garch(N, K, T, S, mean_return, univariate_parameters)

def simulate_garch(N, K, T, S, mean_return, univariate_parameters):
    # Unpack Parameters
    mu = mean_return
    omega = univariate_parameters[:, 0]
    alpha = univariate_parameters[:, 1]
    beta = univariate_parameters[:, 2]

    # Initialize Arrays
    returns = np.zeros((N, K, T, S))
    sigma  = np.zeros((N, K, T, S))
    sigma[:, :, 0, :] = omega / (1 - alpha - beta)
    return returns, sigma