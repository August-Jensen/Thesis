import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.optimize import minimize


# =======================================================
# |                Initail and Basic                    |
# =======================================================

# Create a a numpy array and list of labels from dataframe 
def df_to_array(dataframe):
    """
    Turns a Dataframe into list of column names, and np array of the content

    Takes dataframe

    Creates a list of labels from the columns

    Creates a numpy array of the content

    Returns numpy array, labels
    """
    # Convert the DataFrame to a numpy array
    data_array = df.to_numpy()
    
    # Extract the column names as a list
    labels = df.columns.tolist()
    
    return data_array, labels


# Find the number of parameters in the Correlation Matrix by the number of timeseries
def number_of_corr_params(N):
    """
    Takes the data.shape's N, 

    Returns N*(N-1)/2
    """
    number_of_correlation_parameters = N * (N - 1) / 2
    
    return number_of_correlation_parameters


# Create RSDC_guess, by 2 states plus number of correlation parameters
def initial_RSDC_Guess(number_of_correlation_parameters):
    """
    Generates initial guesses for parameters in an optimization process, including
    transition parameters and correlation parameters.

    Parameters:
    - number_of_correlation_parameters (int): The number    

    Returns:
    - list: A list of initial guesses for the parameters, starting with two transition
            parameters followed by 2 times the number of correlation parameters, each
            with an initial guess of 0.3.

    Example:
    >>> initial_guesses = initial_RSDC_guess(3)
    >>> print(initial_guesses)
    [0.95, 0.95, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
    """
    # Initial guesses for the two transition parameters
    initial_guesses = [0.95, 0.95]
    
    # Ensure number_of_correlation_parameters is an integer
    number_of_correlation_parameters = int(number_of_correlation_parameters)
    
    # # Append initial guesses for the correlation parameters
    # initial_guesses += [0.3] * (2 * number_of_correlation_parameters)
    # return initial_guesses
    
    # Append random initial guesses for the correlation parameters
    # Assuming the random values should be between -1 and 1 for the correlation parameters
    random_guesses = np.random.uniform(0.9, 1, 2 * number_of_correlation_parameters).tolist()
    initial_guesses += random_guesses
    return initial_guesses

# Create Bounds for the RSDC parameters
def set_RSDC_bounds(number_of_correlation_parameters):
    """
    Creates bounds for the RSDC parameters that are to be minimized.

    The first two are for the Transition Probabilities, which are (0.001, 0.999)

    The rest are for the Correlation Parameters, which are (-1, 1)

    Parameters:
    - number_of_correlation_parameters (int): The number of correlation parameters.

    Returns:
    - list of tuples: Each tuple represents the lower and upper bound for a parameter,
                      with the first two parameters (transition probabilities) bounded
                      by (0.001, 0.999) and the correlation parameters bounded by (-1, 1).
    """
    # Ensure number_of_correlation_parameters is treated as an integer
    number_of_correlation_parameters = int(number_of_correlation_parameters)
    
    # Bounds for the two transition parameters
    bounds = [(0.001, 0.999), (0.001, 0.999)]
    
    # Append bounds for the correlation parameters
    bounds += [(-1, 1)] * (2 * number_of_correlation_parameters)
    
    return bounds



# =======================================================
# |             Univariate Model Estimates              |
# =======================================================

# Find the log-likelihood contributions of the univariate volatility
def univariate_log_likelihood_contribution(x, sigma):
    """
    Calculates the Log Likelihood contribution of a univariate GARCH(1,1) in absolute terms

    It first sets sigma to be greater that zero.
    Then calculates the Log-Likelihood contribution as:
        -0.5 * np.log(2 * np.pi) - np.log(sigma) - (x ** 2) / (2 * sigma ** 2)

    Parameters:
    - x (real): The observation of the timeseries, at time t.
    - sigma (real): The value of sigma, at time t.

    Returns:
    - the log_likelihood_contribution
    """
    sigma = max(sigma, 1e-8)
    return -0.5 * np.log(2 * np.pi) - np.log(sigma) - (x ** 2) / (2 * sigma ** 2)


# Calculate the total log-likelihood of the univariate volatility
def total_univariate_log_likelihood(GARCH_guess, x):
    """
    Creates the values of sigma, and then calculates the total_log_likelihood by summing over 
    univariate_log_likelihood_contributions
    
    Parameters:
    - GARCH_guess: omega, alpha and beta from minimize function

    Returns:
    - Negative total_log_likelihood
    """
    
    # Set Number of Observations
    T = len(x)

    # Set Parameters
    omega, alpha, beta = GARCH_guess
    sigma = np.zeros(T)

    # Set the Initial Sigma to be Total Unconditional Variance of data
    sigma[0] = np.sqrt(np.var(x))

    # Calculate sigma[t] for the described model
    for t in range(1, T):
        sigma[t] = omega + alpha * np.abs(x[t-1]) + beta * np.abs(sigma[t-1])

    # Calculate the sum of the Log-Likelihood contributions
    univariate_log_likelihood = sum(univariate_log_likelihood_contribution(x[t], sigma[t]) for t in range(T))

    # Return the Negative Log-Likelihood
    return -univariate_log_likelihood



# Minimize - total log-likelihood of the univariate volatility
def estimate_univariate_models(x):
    """
    Minimizes total_univariate_log_likelihood

    Parameters:
    - data: one dimensional array of time series data.

    Returns:
    - result parameters, and information about accuracy
    
    """
    # Initial Guess for omega, alpha, beta
    GARCH_guess = [0.002, 0.2, 0.7]

    # Minimize the Negative Log-Likelihood Function
    result = minimize(fun=total_univariate_log_likelihood, x0=GARCH_guess, args=(x,), bounds=[(0, None), (0, 1), (0, 1)])
    print(f"Estimated parameters: omega = {result.x[0]}, alpha = {result.x[1]}, beta = {result.x[2]}")

    # Set Parameters
    result_parameters = result.x

    # Set Variance-Covariance Hessian
    result_hessian = result.hess_inv.todense()  

    # Set Standard Errors
    result_se = np.sqrt(np.diagonal(result_hessian))


    # Return Parameters and Information
    return result_parameters, result_hessian, result_se

# Get an array of univariate model parameters for all timeseries
def estimate_univariate_parameters(data):
    """
    Calculates the Univariate estimate for each timeseries in data
    then appends the estimated parameters to estimated_univariate_parameters, 
    and appends the hessian, standard error etc to another list.
    Then it creates a numpy array of these which are returned. 

    Parameters:
    - 

    Returns:
    - 
    
    """
    # Create list to store univariate parameters, hessians, and standard errors
    univariate_parameters = []
    univariate_hessians = []
    univariate_standard_errors = []

    # Iterate over each time series in 'data' and estimate parameters
    for i in range(data.shape[1]):  # data.shape[1] gives the number of time series (columns) in 'data'
        result_parameters, result_hessian, result_se = estimate_univariate_models(data[:, i])
        univariate_parameters.append(result_parameters)
        univariate_hessians.append(result_hessian)
        univariate_standard_errors.append(result_se)

    # Convert the lists of results to numpy arrays
    univariate_parameters_array = np.array(univariate_parameters)
    univariate_hessians_array = np.array(univariate_hessians)
    univariate_standard_errors_array = np.array(univariate_standard_errors)

    # Return the results
    return univariate_parameters_array, univariate_hessians_array, univariate_standard_errors_array




# =======================================================
# |      Functions For Multivariate GARCH Setup         |
# =======================================================

# Forms the Correlation Matrix from RSDC_correlation_guess
def form_correlation_matrix(RSDC_correlation_guess):
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
    n = int(np.sqrt(len(RSDC_correlation_guess) * 2)) + 1
    if len(RSDC_correlation_guess) != n*(n-1)//2:
        raise ValueError("Invalid number of parameters for any symmetric matrix.")
    
    # Create an identity matrix of size n
    matrix = np.eye(n)
    
    # Fill in the off-diagonal elements
    param_index = 0
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i, j] = matrix[j, i] = RSDC_correlation_guess[param_index]
            param_index += 1
            
    return matrix

# Create Parameters for the RSDC Model
def parameterize(RSDC_guess):
    """
    Unfolds the RSDC_guess into Transition Probabilities, and the Parameters for the Correlation Matrix in state 0 and 1
    First, it takes the parameters 0, 1 and sets p_00, p_11 as these.
    Then is separates the rest into 2 lists. 
    It passes these to form_correlation_matrix, which retusn a correlation matrix of the values.
    Finally, it creates an array of the correlation matrix for each state.

    Parameters:
    - RSDC_guess: the list of guesses for the parameters in the model.

    Returns:
    - p_00, p_11: The Transition Probabilities that forms the Transition Matrix
    - correlation...
    
    """
    # Extract Transition Probabilities
    p_00, p_11 = RSDC_guess[0], RSDC_guess[1]

    # Find where to split the parameters, for the remaining parameters.
    split_index = len(RSDC_guess[2:]) // 2

    # Create the arrays of Parameters for Correlation Matrix 0, 1
    correlation_parameters_0 = RSDC_guess[2: 2 + split_index]
    correlation_parameters_1 = RSDC_guess[2 + split_index:]

    # Form the correlation matrix for each state
    correlation_matrix_0 = form_correlation_matrix(correlation_parameters_0)
    correlation_matrix_1 = form_correlation_matrix(correlation_parameters_0)

    # Collect into a single array
    correlation_matrix = [correlation_matrix_0, correlation_matrix_1]

    return p_00, p_11, correlation_matrix



# Create the Trasition Matrix from p_00 & p_11
def create_transition_matrix(p_00, p_11):
    """
    Create the Transition Matrix from p_00 & p_11 with the shape
    p_00, 1-p_11
    1-p_00, p_11

    Returns:
    - transition matrix
    
    """
    transition_matrix = np.zeros([2,2])
    transition_matrix[0] = p_00, 1 - p_11
    transition_matrix[1] = 1 - p_00, p_11

    # Return the Transition Matrix
    return transition_matrix
    



# Calculate Initial State Probabilities by Transition Matrix
def calculate_initial_probabilities(transition_matrix):
    """
    Determine the best guess of the Initial State Probabilities, from Transition Matrix

    Returns:
    - An array of initial probabilities at time t=0
    
    """
    # Needs Comments and expansion
    A_matrix = np.vstack(((np.identity(2)- transition_matrix), np.ones([1,2])))
    pi_first = np.linalg.inv(A_matrix.T.dot(A_matrix)).dot(A_matrix.T)
    pi_second = np.vstack((np.zeros([2,1]), np.ones([1,1])))
    initial_probs = pi_first.dot(pi_second)
    initial_probabilities = initial_probs.T

    return initial_probabilities


# Calculate the Standard Deviations, sigma, from Univariate Estimates
    # This could be done outside of the objective function? 
def calculate_standard_deviations(data, univariate_estimates):
    """
    Calculates the standard deviations, Sigma[t] based on the estimated parameters and the data.


    Parameters:
    - data: The array of data we estimate on.
    - unviariate_estimates: The estimates from the univariate GARCH

    Returns:
    - An array of standard deviations for each timeseries.    
    """
    # Get Data Dimensions
    T, N = data.shape

    # Create Array for Standard Deviations
    standard_deviations = np.zeros((T,N))

    # Calculate Sigmas for each timeseries
    for i in range(N):
        # Unpack Univariate Estimates
        omega, alpha, beta = univariate_estimates[i]

        # Create array for Sigma values
        sigma = np.zeros(T)

        # Set first observation of Sigma to Sample Variance
        sigma[0] = np.sqrt(np.var(data[:, i]))

        # Calculate Sigma[t]
        for t in range(1, T):
            sigma[t] = omega + alpha * np.abs(data[t-1, i]) + beta * np.abs(sigma[t-1])

        # Save Sigmas to Standard Deviation Array
        standard_deviations[:, i] = sigma

    # Return array of all Standard Deviations
    return standard_deviations


# Calculate the Standardized Residuals from Univariate Estimates
    # This could be done outside of the objective function? 
def calculate_standardized_residuals(data, univariate_estimates):
    """
    Calculates the standard deviations, Sigma[t] based on the estimated parameters and the data.


    Parameters:
    - data: The array of data we estimate on.
    - unviariate_estimates: The estimates from the univariate GARCH

    Returns:
    - An array of standard deviations for each timeseries.    
    """
    # Get Data Dimensions
    T, N = data.shape

    # Create Array for Standardized Residuals
    standardized_residuals = np.zeros((T,N))

    # Calculate Sigmas for each timeseries
    for i in range(N):
        # Unpack Univariate Estimates
        omega, alpha, beta = univariate_estimates[i]

        # Create array for Sigma values
        sigma = np.zeros(T)

        # Set first observation of Sigma to Sample Variance
        sigma[0] = np.sqrt(np.var(data[:, i]))

        # Calculate Sigma[t]
        for t in range(1, T):
            sigma[t] = omega + alpha * np.abs(data[t-1, i]) + beta * np.abs(sigma[t-1])

        # Save Sigmas to Standard Deviation Array
        standardized_residuals[:, i] = data[:, i] / sigma

    # Return array of all Standard Deviations
    return standardized_residuals

# Creates a Diagonal Matrix of (N x N), with Standard Deviations on Diagonal, and zeros off the Diagonal

def create_diagonal_matrix(t, std_array):
    """
    Creates an N x N diagonal matrix with standard deviations at time t on the diagonal,
    and zeros elsewhere. Here, N is the number of time series.

    :param t: Integer, the time index for which the diagonal matrix is created.
    :param standard_deviations: List of numpy arrays, each array contains the standard deviations over time for a variable.
    :return: Numpy array, an N x N diagonal matrix with the standard deviations at time t on the diagonal.
    """
    # Extract the standard deviations at time t for each series
    stds_at_t = np.array(std_array[t, :])
    
    # Create a diagonal matrix with these values
    diagonal_matrix = np.diag(stds_at_t)
    
    return diagonal_matrix




# Check if a Correlation Matrix is PSD, Elements in [-1,1], and symmetric.
def check_correlation_matrix_is_valid(correlation_matrix):
    """
    Ensure that the Correlation Matrix satisfies the following:
    1. Diagonal Elements Are 1.
    2. Off-Diagonal Elements are between -1 & 1. 
    3. Check if the Correlation Matrix is Positive Semi-Definite by verifying that the Eigenvalues are non-negative


    Parameters:
    - Correlation Matrix: The Estimated Matrix from the Hamilton Filter

    Returns:
    - Valid: True or False
    - Message: What is not satisfied
    
    """
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



# =======================================================
# |          Hamilton Filter, RSDC Estimation           |
# =======================================================

# Calculates the Log-Likelihood Contribution of the RSDC Model, at time T
def RSDC_log_likelihood_contribution(N, t, data, state, state_correlation_matrix, standard_deviations, standardized_residuals):
    """
    Calculates the likelihood contribution, 

    NOT Log_Likelihood!
    
    Calculates log likelihood contribution,
    Defines D as create_diagonal_matrix(t, standard_deviations)
    Then finds the inverse of D
    And the determinant of D
    It can then calculate the log_likelihood_contribution 
    Then takes the exponential of it, to make filterering step easier.
    

    Parameters:
    - t: Integer of time
    - state: The state that it is calculated at.
    - correlation_matrix, the array of correlation matrix in the states. Selected at state by correlation_matrix[state]

    Returns:
    - 
    """
    # Create D matrix, D_t at time t,
    D = create_diagonal_matrix(t, standard_deviations)

    # # Determinant of D
    # determinant_D = np.linalg.det(D)

    # # Find the Inverse of the state_correlation_matrix
    # # inverse_R = np.linalg.inv(state_correlation_matrix[state])
    # try:
    #     inverse_R = np.linalg.inv(state_correlation_matrix[state])
    # except np.linalg.LinAlgError:
    #     # Apply a small regularization term to the diagonal
    #     eps = 1e-6  # Small positive value
    #     regularized_matrix = state_correlation_matrix[state] + eps * np.eye(state_correlation_matrix[state].shape[0])
    #     inverse_R = np.linalg.inv(regularized_matrix)
    # # Find the Determinant of the state_correlation_matrix
    # determinant_R = np.linalg.det(state_correlation_matrix[state])

    # # Define standardized Residual for a cleaner expression 
    # z_t = standardized_residuals[t]  # z_t at time t
        
    # # Log likelihood contribution for time t
    # # ///XXX Here Was an Error!
    # log_likelihood_contribution = -0.5 * (N * np.log(2 * np.pi) + 2 * np.log(determinant_D) + np.log(determinant_R) + z_t.T @ inverse_R @ z_t)
    inv_D = np.linalg.inv(D)
    det_D = np.linalg.det(D)

    R = state_correlation_matrix[state]
    
    # Calculate H_t, the conditional covariance matrix
    H_t = D @ R @ D

    # Compute the likelihood contribution for this time t
    inv_H_t = np.linalg.inv(H_t)  # Inverse of H_t
    det_H_t = np.linalg.det(H_t)  # Determinant of H_t
    z_t = inv_D @ data[t]  # z_t at time t
    
    # Log likelihood contribution for time t
    log_likelihood_contribution = -0.5 * (N * np.log(2 * np.pi) + 2 * np.log(det_D)  + z_t.T @ inv_H_t @ z_t)
    #print(f' The Log Likelihood: {log_likelihood_contribution} \n The Exponential: {np.exp(log_likelihood_contribution)}')
    return log_likelihood_contribution


# Calculates the Total Log Likelihood of the RSDC Model
    # Including Predicted Probability and Filtered Probability
def RSDC_total_log_likelihood(data, RSDC_guess, standard_deviations, standardized_residuals):
    """
    This Function is Minimized in the fit function.
    It should 
    """
    # Get Shape of Data
    T, N = data.shape
    n_states = 2
    # Array for Predicted Probabilities
    predicted_probabilities = np.zeros([n_states, T + 1])
    
    # Array for Filtered Probabilities
    filtered_probabilities = np.zeros([n_states, T])

    # Array for Log-Likelihood Contributions
    log_likelihood_contributions = np.zeros(T)

    # Form Model Parameters (With State Correlation Matrix)
    p_00, p_11, state_correlation_matrix = parameterize(RSDC_guess)

    # Form Transition Matrix
    transition_matrix = create_transition_matrix(p_00, p_11)

    # Form Initial Probabilities, predicted Probabilities at time t=0
    predicted_probabilities[[0,1],0] = calculate_initial_probabilities(transition_matrix)
    #print(predicted_probabilities[[0,1],0])
    
    # To hold values of RSDC_likelihood_contributions
    eta = np.zeros(n_states)

    # The Partila RSDC Likelihood Contributions
    partial_likelihood = np.zeros(n_states)

    # The Hamilton Filter Loop
    for t in range(T):
        
        # Calculate the state Densities, Eta and the Partial Likelihoods
        for state in range(n_states):
            eta[state] = RSDC_log_likelihood_contribution(N, t, data, state, state_correlation_matrix, standard_deviations, standardized_residuals)
            # Applying the log-sum-exp trick
        M = np.max(eta)
        log_sum_exp = M + np.log(np.sum(np.exp(eta - M)))
        # Compute log(L0 / (L0 + L1))
        log_fraction_L0 = eta[0] - log_sum_exp
        log_fraction_L1 = eta[1] - log_sum_exp
        eta[0] = np.exp(log_fraction_L0)
        eta[1] = np.exp(log_fraction_L1)

        # Now, to use this in normalization:
        normalized_log_likelihoods = eta - log_sum_exp
        #print('Norm', normalized_log_likelihoods)
        # For comparison, let's also calculate the direct way which should result in underflow
        #direct_exp_normalization = np.exp(eta) / np.sum(np.exp(eta))
        
        #(log_sum_exp, normalized_log_likelihoods, direct_exp_normalization)
        
        for state in range(n_states):
            partial_likelihood[state] = predicted_probabilities[state,t] * normalized_log_likelihoods[state]
        # print(f'Eta: {eta[state]}')
        #print(eta,partial_likelihood)
        # Calculate the log_likelihood_contribution
        # ///xxx Error! Changed np.log(np.sum(partial_likelihood)) to np.sum(np.log(partial_likelihood))
        log_likelihood_contributions[t] = np.sum(partial_likelihood)

        #Filtering Step
        num0 = eta[0] * predicted_probabilities[0,t]
        num1 = eta[1] * predicted_probabilities[1,t]
        denom = num0 + num1
        filter0 = num0 / denom
        filter1 = num1 / denom
        # print(f'FIlter 1: {filter1}')
        filtered_probabilities[[0,1],t] = filter0, filter1

        # Prediction Step
        predicted_probabilities[[0,1],t+1] = transition_matrix.dot(filtered_probabilities[[0,1],t])
        # print(f'eta:  {eta}')
        # print(f'partial_likelihood:  {partial_likelihood}')
        # print(f'predicted_probabilities:  {predicted_probabilities[[0,1],t]}')
        # print(f'filtered_probabilities:  {filtered_probabilities[[0,1],t]}')
    #Find the Negative Total Log Likelihood
    negative_likelihood = - np.sum(log_likelihood_contributions)
    # print(negative_likelihood)
    # Return Negative Likelihood
    return negative_likelihood



# Minimize the Models Objective Function
    # Remember to Get params, hess_inf.to_dense(), as well as possibly fun, jac, and nit.
def fit(dataframe):
    """
    This function minimizes the negative Log-Likelihood
    1. Create an array of the data.
    2. Set T, N
    3. Estimate GARCH
    4. Set Number of Correlation Parameters
    5. Set RSDC Guess
    6. Set RSDC Bounds
    7. Define Objective Function
    8. Minimize
    9. Get Results
    10. Check PSD
    ...
    """

    # Create Data & Labels
    data, labels = df_to_array(dataframe)

    # Get Shape
    T, N = data.shape

    # List of Results

    # Estimate GARCH
    univariate_parameters_array, univariate_hessians_array, univariate_standard_errors_array = estimate_univariate_parameters(data)
    
    # Set Number of Correlation Parameters
    number_of_correlation_parameters = number_of_corr_params(N)

    # Set Initial RSDC Guess
    RSDC_guess = initial_RSDC_Guess(number_of_correlation_parameters)

    # Calculate Standard Deviations & Standardized Residuals
    standard_deviations = calculate_standard_deviations(data, univariate_parameters_array)
    standardized_residuals = calculate_standardized_residuals(data, univariate_parameters_array)

    # Set Initial RSDC Bounds
    RSDC_Bounds = set_RSDC_bounds(number_of_correlation_parameters)
    print('Guess', RSDC_guess)#, params)
    # Inside Fit or Inside RSDC Likelihood?
    def objective_function(RSDC_guess):
        return RSDC_total_log_likelihood(data, RSDC_guess, standard_deviations, standardized_residuals)
    result = minimize(objective_function, RSDC_guess, bounds=RSDC_Bounds, method='L-BFGS-B')

    if result.success:
        # Print Sucessful Optimization
        print("Optimization was successful.")
        # Store Results
        RSDC_params = result.x
        RSDC_hessian = result.hess_inv.todense()
        RSDC_se = np.sqrt(np.diagonal(RSDC_hessian))
        RSDC_fun = result.fun
        RSDC_jac = result.jac
        RSDC_nit = result.nit

        # Create Dictionary of Results
        results = {
            'labels': labels,
            'univariate_parameters_array': univariate_parameters_array,
            'univariate_hessians_array': univariate_hessians_array,
            'univariate_standard_errors_array': univariate_standard_errors_array,
            'RSDC_params': RSDC_params,
            'RSDC_hessian': RSDC_hessian,
            'RSDC_se': RSDC_se,
            'fun': RSDC_fun,
            'jac': RSDC_jac,
            'nit': RSDC_nit
        }
        print(f'The Estimated Parameters: \n {RSDC_params}')
        # Find where to split the parameters, for the remaining parameters.
        split_index = len(RSDC_guess[2:]) // 2
    
        # Create the arrays of Parameters for Correlation Matrix 0, 1
        correlation_parameters_0 = RSDC_params[2: 2 + split_index]
        correlation_parameters_1 = RSDC_params[2 + split_index:]
    
        # Form the correlation matrix for each state
        correlation_matrix_0 = form_correlation_matrix(correlation_parameters_0)
        correlation_matrix_1 = form_correlation_matrix(correlation_parameters_0)
    
        # Collect into a single array
        estimated_correlation_matrix = [correlation_matrix_0, correlation_matrix_1]
        print(f'The Estimated Correlation Matrix: \n {estimated_correlation_matrix}')
        valid, message = check_correlation_matrix_is_valid(estimated_correlation_matrix)
        print(message)
            
    else:
        print("Optimization failed.")
        return result


