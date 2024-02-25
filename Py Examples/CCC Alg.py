

        # Create the D_t matrix for this time t
        D = create_diagonal_matrix(t, standard_deviations)
        inv_D = np.linalg.inv(D)
        det_D = np.linalg.det(D)

        # Calculate H_t, the conditional covariance matrix
        H_t = D @ R @ D
        
        # Compute the likelihood contribution for this time t
        inv_H_t = np.linalg.inv(H_t)  # Inverse of H_t
        det_H_t = np.linalg.det(H_t)  # Determinant of H_t
        z_t = inv_D @ data[t]  # z_t at time t
        
        # Log likelihood contribution for time t
        log_likelihood_contribution = -0.5 * (N * np.log(2 * np.pi) + 2 * np.log(det_D)  + z_t.T @ inv_H_t @ z_t)











# Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns
# Tickers & Period
# tickers = ['ACWI', 'SCZ', 'SPY', 'SHY', 'TLT', 'AAPL', 'TSLA']

# tickers = [
#     'ACWI',  # MSCI All Country World Index
#     'SCZ',   # iShares MSCI EAFE Small-Cap ETF
#     'SPY',   # SPDR S&P 500 ETF Trust
#     # 'SHY',   # iShares 1-3 Year Treasury Bond ETF
#     'TLT',   # iShares 20+ Year Treasury Bond ETF
#     'XLF',   # Financial Select Sector SPDR Fund
#     'XLK',   # Technology Select Sector SPDR Fund
#     'VWO',   # Vanguard FTSE Emerging Markets ETF
# #    'JNK'    # SPDR Bloomberg Barclays High Yield Bond ETF (High Yield Bond)

# ]

# labels = [
#     'ACWI - MSCI All Country World Index ETF',
#     'SCZ - iShares MSCI EAFE Small-Cap ETF',
#     'SPY - SPDR S&P 500 ETF Trust',
#     # 'SHY - iShares 1-3 Year Treasury Bond ETF',
#     'TLT - iShares 20+ Year Treasury Bond ETF',
#     'XLF - Financial Select Sector SPDR Fund',
#     'XLK - Technology Select Sector SPDR Fund',
#     'VWO - Vanguard FTSE Emerging Markets ETF',
# #    'JNK - SPDR Bloomberg Barclays High Yield Bond ETF'
# ]

tickers = [
    'MAJDKO.CO',
    'MAJGO.CO',
    '',
    '',
    '',
    '',
    '',
    '',
    '',
    '',
    '',
    '',
]

labels = [
    'Maj Dk Obl',
    'Maj Gl Obl',
    '',
    '',
    '',
    '',
    '',
    '',
    '',
    '',
    '',
    '',
    
]

start_date = '2020-01-01'
end_date = '2024-01-01'

# Dictionary of Closing Prices
closing_prices = {}

for ticker in tickers:
    data = yf.download(ticker, start=start_date, end=end_date)
    closing_prices[ticker] = data['Close']

# Make a DataFrame out of Dictionary
price_frame = pd.DataFrame(closing_prices)

# Get Log prices
log_prices = np.log(price_frame)

# Get log difference of data
log_returns = log_prices.diff().dropna()
data_df = log_returns
# Get Numpy Array of the Data
data = log_returns.to_numpy()

# Print Results
log_returns.head(), data.shape


# Plotting the line plot of the simulated stock data
plt.figure(figsize=(14, 7))
sns.lineplot(data=data, palette="pastel", dashes=False)
plt.title('Simulated Log Returns of Stocks')
plt.xlabel('Days')
plt.ylabel('Log Returns')
plt.legend(title='Ticker')
plt.savefig('lineplot.png')  # Saves the line plot
plt.show()

# Plotting the heatmap of the correlations
correlation_matrix = data_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5, xticklabels=labels, yticklabels=labels)
plt.title('Correlation Matrix of Log Returns')
plt.savefig('unmodelled_heatmap.png') 
plt.show()

# Step 1: Calculate variance for each column to determine volatility
variances = data_df.var()

# Step 2: Sort the DataFrame based on variance (volatility), descending order
sorted_columns = variances.sort_values(ascending=False).index
data_sorted = data_df[sorted_columns]
# Calculate variance for each column to determine volatility


# Plotting the line plot of the sorted simulated stock data
plt.figure(figsize=(14, 7))

# Create a color palette with a distinct color for each series
palette = sns.color_palette("pastel", len(sorted_columns))

# Plot each series individually to ensure correct color and legend handling
for i, column in enumerate(sorted_columns):
    sns.lineplot(x=data_sorted.index, y=data_sorted[column], color=palette[i], label=column)

plt.title('Simulated Log Returns of Stocks (Sorted by Volatility)')
plt.xlabel('Days')
plt.ylabel('Log Returns')
plt.legend(title='Ticker')
plt.savefig('lineplot_sorted_by_volatility.png')  # Saves the sorted line plot
plt.show()

# Calculate the cumulative sum of the series for each stock
data_cumsum = data_sorted.cumsum()

# Plotting the cumulative sum
plt.figure(figsize=(14, 7))

# Plot each series in the cumulative sum DataFrame individually
for i, column in enumerate(sorted_columns):
    sns.lineplot(x=data_cumsum.index, y=data_cumsum[column], color=palette[i], label=column)

plt.title('Cumulative Log Returns of Stocks')
plt.xlabel('Days')
plt.ylabel('Cumulative Log Returns')
plt.legend(title='Ticker')
plt.savefig('cumulative_log_returns.png')  # Saves the cumulative sum plot
plt.show()


from scipy.optimize import minimize

# The Likelihood Contribution Function
def log_likelihood_contribution(x, sigma):
    sigma = max(sigma, 1e-8)
    return -0.5 * np.log(2 * np.pi) - np.log(sigma) - (x ** 2) / (2 * sigma ** 2)

# The Total log Likelihood
def total_log_likelihood(parameters, x):
    T = len(x)
    omega, alpha, beta = parameters
    sigma = np.zeros(T)
    # Set initial Sigma to total variance of data
    sigma[0] = np.sqrt(np.var(x))

    # Calculate sigma[t] based on the described model
    for t in range(T):
        sigma[t] = omega + alpha * np.abs(x[t-1]) + beta * np.abs(sigma[t-1])

    # Calculate the log likelihood contributions, and find sum.
    log_likelihood = sum(log_likelihood_contribution(x[t], sigma[t]) for t in range(T))

    # Return Negative Total Log Likelihood
    return - log_likelihood

# Estimate Model
def estimate_garch_parameters(x):
    # Initial Guess of Parameters
    initial_parameters = [0.05, 0.15, 0.85]
    
    # Minimize the Negative Log-Likelihood
    result = minimize(fun=total_log_likelihood, x0=initial_parameters, args=(x,), bounds=[(0, None), (0, 1), (0, 1)])

    if result.success:
        print("Optimization was successful.")
        print(f"Estimated parameters: omega = {result.x[0]}, alpha = {result.x[1]}, beta = {result.x[2]}")
        return result.x
    else:
        print("Optimization failed.")
        return [None, None, None]   

# Initialize a list to store estimation results
estimation_results = []

# Iterate over each time series in 'data' and estimate parameters
for i in range(data.shape[1]):  # data.shape[1] gives the number of time series (columns) in 'data'
    params = estimate_garch_parameters(data[:, i])
    estimation_results.append(params)

# Convert the list of results to a numpy array
estimation_results_array = np.array(estimation_results)

estimation_results_array






import numpy as np
from scipy.optimize import minimize
from numpy.linalg import det, inv

def calculate_std(data,univariate):
    T, N = data.shape
    standard_deviations = np.zeros((T,N))
    for i in range(N):
        omega, alpha, beta = univariate[i]
        sigma = np.zeros(T)
        # Set first observation to sample variance
        sigma[0] = np.sqrt(np.var(data[:, i]))
        for t in range(1, T):
            sigma[t] = omega + alpha * np.abs(data[t-1, i]) + beta * np.abs(sigma[t-1])
        standard_deviations[:,i] = sigma
    return standard_deviations

def create_diagonal_matrix(t, my_array):
    """
    Creates an N x N diagonal matrix with standard deviations at time t on the diagonal,
    and zeros elsewhere. Here, N is the number of time series.

    :param t: Integer, the time index for which the diagonal matrix is created.
    :param standard_deviations: List of numpy arrays, each array contains the standard deviations over time for a variable.
    :return: Numpy array, an N x N diagonal matrix with the standard deviations at time t on the diagonal.
    """
    # Extract the standard deviations at time t for each series
    stds_at_t = np.array(my_array[t, :])
    
    # Create a diagonal matrix with these values
    diagonal_matrix = np.diag(stds_at_t)
    
    return diagonal_matrix

def calculate_standardized_residuals(data, univariate):
    T, N = data.shape
    standardized_residuals = np.zeros((T, N))
    for i in range(N):
        omega, alpha, beta = univariate[i]
        sigma = np.zeros(T)
        sigma[0] = np.sqrt(np.var(data[:, i]))  # Initialize with sample variance
        for t in range(1, T):
            sigma[t] = omega + alpha * np.abs(data[t-1, i]) + beta * np.abs(sigma[t-1])
        standardized_residuals[:, i] = data[:, i] / sigma
    return standardized_residuals

# Rest of Parameterize
def parameterize(params):
    """
    Takes parameters and creates:
        p_00, p_11: Transition probabilities
        corr_params_0, corr_params_1: The two halves of the remaining parameters

    Parameters:
        params (list or np.array): Input parameters including p_00, p_11 and correlational parameters.

    Returns:
        p_00 (float): Transition probability for state 0 to stay in state 0.
        p_11 (float): Transition probability for state 1 to stay in state 1.
        corr_params_0 (np.array): First half of the correlational parameters.
        corr_params_1 (np.array): Second half of the correlational parameters.
    """
    # Extract p_00 and p_11
    p_00, p_11 = params[0], params[1]
    
    # Calculate the split index for the remaining parameters
    split_index = len(params[2:]) // 2
    
    # Split the remaining parameters into two equal halves
    corr_params_0 = params[2: 2 + split_index]
    corr_params_1 = params[2 + split_index:]
    
    return p_00, p_11, corr_params_0, corr_params_1








# Check if the estimated correlation matrix is positive semi-definite
def is_positive_semi_definite(matrix):
    return np.all(np.linalg.eigvals(matrix) >= -1e-10)  # Allow for numerical precision issues

def form_corr_matrix(params):
    """
    Creates a square matrix with ones on the diagonal and symmetric off-diagonal elements
    based on the input list of parameters.
    
    Parameters:
    - params: A list of numbers to fill into the off-diagonal elements. The length of this list
              should be n(n-1)/2 for a square matrix of size n.
    
    Returns:
    - A numpy array representing the square matrix with the specified properties.
    """
    # Determine the size of the matrix
    n = int(np.sqrt(len(params) * 2)) + 1
    if len(params) != n*(n-1)//2:
        raise ValueError("Invalid number of parameters for any symmetric matrix.")
    
    # Create an identity matrix of size n
    matrix = np.eye(n)
    
    # Fill in the off-diagonal elements
    param_index = 0
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i, j] = matrix[j, i] = params[param_index]
            param_index += 1
            
    return matrix


def CCC_log_likelihood(data, univariate, parameters):
    T, N = data.shape  # Number of observations and time series
    
    # Assuming calculate_standardized_residuals and calculate_std are already defined
    standardized_residuals = calculate_standardized_residuals(data, univariate)
    standard_deviations = calculate_std(data, univariate)
    
    # Form the correlation matrix R using the remaining parameters
    # Assuming the last N*(N-1)/2 parameters are for the correlation matrix
    R = form_corr_matrix(parameters[-int(N*(N-1)/2):])
    
    total_log_likelihood = 0
    
    for t in range(T):
        # Create the D_t matrix for this time t
        D = create_diagonal_matrix(t, standard_deviations)
        inv_D = np.linalg.inv(D)
        det_D = np.linalg.det(D)
        # Calculate H_t, the conditional covariance matrix
        # H_t = D_t @ R @ D_t
        
        # Compute the likelihood contribution for this time t
        # inv_H_t = np.linalg.inv(H_t)  # Inverse of H_t
        # det_H_t = np.linalg.det(H_t)  # Determinant of H_t
        z_t = standardized_residuals[t]  # z_t at time t
        
        # Log likelihood contribution for time t
        log_likelihood_contribution = -0.5 * (N * np.log(2 * np.pi) + np.log(det_D) + z_t.T @ inv_D @ z_t)
        total_log_likelihood += log_likelihood_contribution
    
    return -total_log_likelihood  # Return negative for minimization

def optimize_CCC_model(data):
    T, N = data.shape  # Number of observations and number of time series
    initial_guess = np.random.rand(N * (N - 1) // 2)  # Initial guess for correlations
    print('Guess', initial_guess, params)
    bounds = [(-1, 1) for _ in range(N * (N - 1) // 2)]  # Bounds for correlations
    
    # Define a wrapper function for CCC_log_likelihood to include only parameters as variable
    def objective_function(params):
        return CCC_log_likelihood(data, estimation_results_array, params)
    
    result = minimize(objective_function, initial_guess, bounds=bounds, method='L-BFGS-B')
    
    if result.success:
        optimized_params = result.x
        print("Optimization was successful.")
        return optimized_params
    else:
        print("Optimization failed.")
        return None






# Calculate the standardized residuals
standardized_residuals = calculate_standardized_residuals(data, estimation_results_array)
standard_deviations = calculate_std(data, estimation_results_array)
diag = create_diagonal_matrix(1, standard_deviations)
z = np.linalg.inv(diag) * data[1].T
y = np.linalg.inv(diag) * data[1]
z,standardized_residuals[1],diag
# fit = optimize_CCC_model(data, estimation_results_array)
fit = optimize_CCC_model(data)
estimated_matrix = form_corr_matrix(fit)
print(estimated_matrix)


# Plotting the heatmap of the correlations
plt.figure(figsize=(10, 8))
sns.heatmap(estimated_matrix, annot=True, cmap='coolwarm', linewidths=.5, xticklabels=labels, yticklabels=labels)
plt.title('Constant Conditional Correlation Matrix')
plt.savefig('CCC_Heatmap.png')
plt.show()



def is_valid_correlation_matrix(matrix):
    # Check diagonal elements are all 1
    if not np.all(np.diag(matrix) == 1):
        return False, "Not all diagonal elements are 1."
    
    # Check off-diagonal elements are between -1 and 1
    if not np.all((matrix >= -1) & (matrix <= 1)):
        return False, "Not all off-diagonal elements are between -1 and 1."
    
    # Check if the matrix is positive semi-definite
    # A matrix is positive semi-definite if all its eigenvalues are non-negative.
    eigenvalues = np.linalg.eigvals(matrix)
    if np.any(eigenvalues < -0.5):
        print(eigenvalues)
        return False, "The matrix is not positive semi-definite."
    
    return True, "The matrix meets all criteria."


valid, message = is_valid_correlation_matrix(estimated_matrix)
print(message)


valid, message = is_valid_correlation_matrix(correlation_matrix)
print(message)<