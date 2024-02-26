import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

def df_to_array(dataframe):
    # Create Numpy Array
    data_array = df.to_numpy().T
    

    # Get titles of columns for plotting
    labels = df.columns.tolist()

    return data_array, labels

# Find the log-likelihood contributions of the univariate volatility
def univariate_log_likelihood_contribution(x, sigma):
    sigma = max(sigma, 1e-8)
    return -0.5 * np.log(2 * np.pi) - np.log(sigma) - (x ** 2) / (2 * sigma ** 2)


# Calculate the total log-likelihood of the univariate volatility
def total_univariate_log_likelihood(GARCH_guess, x):
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
    # Initial Guess for omega, alpha, beta
    GARCH_guess = [0.002, 0.2, 0.7]

    # Minimize the Negative Log-Likelihood Function
    result = minimize(fun=total_univariate_log_likelihood, x0=GARCH_guess, args=(x,), bounds=[(0, None), (0, 1), (0, 1)])
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
def estimate_univariate_parameters(data, labels):
    # Create list to store univariate parameters, hessians, and standard errors
    univariate_parameters = []
    # univariate_hessians = []
    # univariate_standard_errors = []

    # Iterate over each time series in 'data' and estimate parameters
    for i in range(data.shape[0]):  # data.shape[1] gives the number of time series (columns) in 'data'
        result_parameters, result_hessian, result_se = estimate_univariate_models(data[:, i])
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

    
# Forms the Correlation Matrix from RSDC_correlation_guess
def form_correlation_matrix(multi_guess):
    # Determine the size of the matrix
    n = int(np.sqrt(len(multi_guess) * 2)) + 1
    if len(multi_guess) != n*(n-1)//2:
        raise ValueError("Invalid number of parameters for any symmetric matrix.")
    
    # Create an identity matrix of size n
    matrix = np.eye(n)
    
    # Fill in the off-diagonal elements
    param_index = 0
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i, j] = matrix[j, i] = multi_guess[param_index]
            param_index += 1
            
    return matrix


# Calculate the Standard Deviations, sigma, from Univariate Estimates
    # This could be done outside of the objective function? 
def calculate_standard_deviations(data, univariate_estimates):
    # Get Data Dimensions
    N,T = data.shape

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
            sigma[t] = omega + alpha * np.abs(data[i,t-1]) + beta * np.abs(sigma[t-1])

        # Save Sigmas to Standard Deviation Array
        standard_deviations[:, i] = sigma

    # Return array of all Standard Deviations
    return standard_deviations


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
    stds_at_t = np.array(std_array[t,:])
    
    # Create a diagonal matrix with these values
    diagonal_matrix = np.diag(stds_at_t)
    
    return diagonal_matrix




# Check if a Correlation Matrix is PSD, Elements in [-1,1], and symmetric.
def check_correlation_matrix_is_valid(correlation_matrix):
    # Check diagonal elements are all 1
    if not np.all(np.diag(correlation_matrix) == 1):
        return False, "Not all diagonal elements are 1."
    
    # Check off-diagonal elements are between -1 and 1
    if not np.all((correlation_matrix >= -1) & (correlation_matrix <= 1)):
        return False, "Not all off-diagonal elements are between -1 and 1."
    
    # Check if the matrix is positive semi-definite
    # A matrix is positive semi-definite if all its eigenvalues are non-negative.
    eigenvalues = np.linalg.eigvals(correlation_matrix)
    if np.any(eigenvalues < -0.5):
        print(eigenvalues)
        return False, "The matrix is not positive semi-definite."
    
    return True, "The matrix meets all criteria."




   # # Calculate the correlation matrix from the data
   #  corr_matrix = np.corrcoef(data, rowvar=False)
    
   #  # Extract the off-diagonal elements of the correlation matrix as parameters
   #  off_diagonal_elements = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
    
   #  # Define the number of correlation parameters as the length of the off-diagonal elements
   #  number_of_correlation_parameters = len(off_diagonal_elements)
    
   #  # Initialize the parameters list with the first two elements
   #  parameters = [0.95, 0.95]
    
   #  # Use the correlations as uniform_randoms
   #  uniform_randoms = off_diagonal_elements
    
   #  # Take half the value of these for random_between
   #  random_between = uniform_randoms / 2
    
   #  # Extend the parameters list with uniform_randoms and random_between
   #  parameters.extend(uniform_randoms)
   #  parameters.extend(random_between)
    
   #  return parameters

def initial_parameters(number_of_correlation_parameters):

    # Ensure the input is treated as an integer
    number_of_correlation_parameters = int(number_of_correlation_parameters)
    
    # First two elements are 0.95 each
    parameters = [0.95, 0.95]
    
    # Next set of elements, randomly chosen in uniform [0, 0.9]
    uniform_randoms = np.random.uniform(0, 0.4, number_of_correlation_parameters)
    parameters.extend(uniform_randoms)
    
    # Last set of elements, randomly chosen between [-0.5, 0.9]
    random_between = np.random.uniform(-0.1, 0.4, number_of_correlation_parameters)
    parameters.extend(random_between)
    
    return parameters
def set_bounds(number_of_correlation_parameters):
    # Ensure the input is treated as an integer
    number_of_correlation_parameters = int(number_of_correlation_parameters)
    
    # First two bounds are (0.01, 0.99)
    bounds = [(0.01, 0.99), (0.01, 0.99)]
    
    # The rest are 2 * number_of_correlation_parameters of bounds (-0.99, 0.99)
    bounds.extend([(-0.99, 0.99) for _ in range(2 * number_of_correlation_parameters)])
    
    return bounds

def parameterize(RSDC_guess):
    # Extract Transition Probabilities
    p_00, p_11 = RSDC_guess[0], RSDC_guess[1]

    # Find where to split the parameters, for the remaining parameters.
    split_index = len(RSDC_guess[2:]) // 2

    # Create the arrays of Parameters for Correlation Matrix 0, 1
    correlation_parameters_0 = RSDC_guess[2: 2 + split_index]
    correlation_parameters_1 = RSDC_guess[2 + split_index:]

    # Form the correlation matrix for each state
    correlation_matrix_0 = form_correlation_matrix(correlation_parameters_0)
    correlation_matrix_1 = form_correlation_matrix(correlation_parameters_1)

    # Collect into a single array
    correlation_matrix = [correlation_matrix_0, correlation_matrix_1]
    correlation_matrix = np.array(correlation_matrix)

    return p_00, p_11, correlation_matrix


def parameterize_results(param_estimate):
    # Extract Transition Probabilities
    p_00, p_11 = param_estimate[0], param_estimate[1]

    # Find where to split the parameters, for the remaining parameters.
    split_index = len(param_estimate[2:]) // 2

    # Create the arrays of Parameters for Correlation Matrix 0, 1
    correlation_parameters_0 = param_estimate[2: 2 + split_index]
    correlation_parameters_1 = param_estimate[2 + split_index:]

    # Form the correlation matrix for each state
    correlation_matrix_0 = form_correlation_matrix(correlation_parameters_0)
    correlation_matrix_1 = form_correlation_matrix(correlation_parameters_1)

    # Collect into a single array
    correlation_matrix = [correlation_matrix_0, correlation_matrix_1]
    correlation_matrix = np.array(correlation_matrix)

    return p_00, p_11, correlation_matrix


def create_transition_matrix(p_00, p_11):
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


def ccc_likelihood_contribution(t, data, R, standard_deviations):
    # What we need in the terms:
    data = data.T
    D = create_diagonal_matrix(t, standard_deviations)
    # R is defined in Total CCC Likelihood 
    
    # Linear Algebra
    det_D = np.linalg.det(D)
    if det_D <1e-8:
        det_D = 1e-6
    inv_D = np.linalg.inv(D)
    det_R = np.linalg.det(R)
    if det_R < 1e-8:
        punish = det_R
        det_R = 1e-6
    else:
        punish = 0
    # inv_R = np.linalg.inv(R)
    inv_R = np.linalg.inv(R) if det_R >= 0 else np.eye(R.shape[0])  # Fallback for negative det_R
    
    lambda_penalty = 1000
   
    # Penalty for negative det_R
    penalty = lambda_penalty * min(0, punish)  # This will be non-zero only if det_R is negative

    # The Shock Term
    z = inv_D @ data[t]

    # The Terms of the Log Likelihood Contribution
    term_1 = N * np.log(2 * np.pi)
    term_2 = 2 * np.log(det_D) 
    term_3 = np.log(det_R)
    term_4 = z.T @ inv_R @ z
    
    log_likelihood_contribution = -0.5 * (term_1 + term_2 + term_3 + term_4) + penalty
    
    return np.exp(log_likelihood_contribution)

# def Hamilton_Filter(data,random_guesses, standard_deviations):
#     # Get Shape of Data
#     N, T = data.shape

#     # Set n_states to 2, to make expanding to more states easier
#     n_states = 2

#     # Array for Predicted Probabilities
#     predicted_probabilities = np.zeros([n_states, T + 1])
    
#     # Array for Filtered Probabilities
#     filtered_probabilities = np.zeros([n_states, T])
#     # print(filtered_probabilities.shape)
#     # Array for Log-Likelihood Contributions
#     log_likelihood_contributions = np.zeros(T)

#     # Get Transition Probabilities & R matrix
#     p_00, p_11, R = parameterize(random_guesses)

#     #Create The Transition Matrix
#     transition_matrix = create_transition_matrix(p_00, p_11)
    
#     # Set initial_probability by Transition Matrix
#     predicted_probabilities[[0,1],0] = calculate_initial_probabilities(transition_matrix)
#     # print(predicted_probabilities[[0,1],0])

#     # Array for Log-Likelihoods Contributions
#     log_likelihood_contributions = np.zeros(T)
#     # To hold values of RSDC_likelihood_contributions
#     eta = np.zeros(n_states)

#     # The Partila RSDC Likelihood Contributions
#     partial_likelihood = np.zeros(n_states)
    
#     # The Hamilton Filter Loop
#     for t in range(T):
#         # Calculate the state Densities, Eta and the Partial Likelihoods
#         for state in range(n_states):
#             # At this stage Eta is log.
#             corr_mat = R[state]
            
#             eta[state] = ccc_likelihood_contribution(t, data, corr_mat, standard_deviations)
#             # partial_likelihood[state] = predicted_probabilities[state,t] * eta[state]
#         # Applying the log-sum-exp trick
#         M = np.max(eta)
#         # At this stage Eta is log
#         log_sum_exp = M + np.log(np.sum(np.exp(eta - M)))
#         # print(log_sum_exp)
#         #Compute log(L0 / (L0 + L1))
        
#         # At this stage Eta is log
#         log_fraction_L0 = eta[0] - log_sum_exp
#         log_fraction_L1 = eta[1] - log_sum_exp
        
#         # print(log_fraction_L0)
        
#         # At this stage Eta is normal
#         eta[0] = log_fraction_L0
#         eta[1] = log_fraction_L1
#         # Now, to use this in normalization:
#         # normalized_log_likelihoods = eta - log_sum_exp
#         #print('Norm', normalized_log_likelihoods)
#         # For comparison, let's also calculate the direct way which should result in underflow
#         #direct_exp_normalization = np.exp(eta) / np.sum(np.exp(eta))
        
#         #(log_sum_exp, normalized_log_likelihoods, direct_exp_normalization)
        
#         # for state in range(n_states):
#         #     partial_likelihood[state] = predicted_probabilities[state,t] * eta[state]

#         log_predicted_probabilities = np.log(predicted_probabilities[:, t])
        
#         # Assuming eta is already adjusted for each state's likelihood contribution
#         weighted_log_likelihoods = eta + log_predicted_probabilities
        
#         # Step 2: Apply the log-sum-exp trick to the weighted log likelihoods
#         M = np.max(weighted_log_likelihoods)
#         log_sum_exp = M + np.log(np.sum(np.exp(weighted_log_likelihoods - M)))
        
#         # The result is the log likelihood contribution for time t
#         log_likelihood_contributions[t] = log_sum_exp
#         #log_likelihood_contributions[t] = np.log(np.sum(partial_likelihood))

#         filtered_probabilities[[0,1],t] = np.exp(log_fraction_L0), np.exp(log_fraction_L1)
#         # #Filtering Step
#         # num0 = eta[0] * predicted_probabilities[0,t]
#         # num1 = eta[1] * predicted_probabilities[1,t]
#         # denom = num0 + num1
#         # filter0 = num0 / denom
#         # filter1 = num1 / denom
#         # filtered_probabilities[[0,1],t] = filter0, filter1

#         # Prediction Step
#         predicted_probabilities[[0,1],t+1] = transition_matrix.dot(filtered_probabilities[[0,1],t])

    
#     # print(f'eta:  {eta}')
#     # print(f'partial_likelihood:  {partial_likelihood}')
#     # print(f'predicted_probabilities:  {predicted_probabilities}')
#     # print(f'filtered_probabilities:  {filtered_probabilities}')
    
#     #Find the Negative Total Log Likelihood
#     negative_likelihood = - np.sum(log_likelihood_contributions)
#     # print(negative_likelihood)

    
#     # Return Negative Likelihood
#     return negative_likelihood
# # standard_deviations = np.zeros((N,T))
# # standard_deviations = calculate_standard_deviations(data, univ_params)
# # guess = [0.95, 0.95, 0.03026404976969012, 0.2599401568540691, 0.4424802671431618, 0.2956209785258734, 0.0789336159306628, 0.3740201925493622, 0.20507063250295035, 0.157726295631511, 0.05272054968913, 0.219647907330339, -0.38049635093756246, 0.11810087419442217, 0.49998028364185887, 0.05252799553877302, 0.34406057578824767, 0.1002056297127, 0.17737394702802894, -0.097594140473587, 0.2962655755884279, 0.2268062811449]
# # Hamilton_Filter(data, guess,standard_deviations)


def Hamilton_Filter(data, random_guesses, standard_deviations):
    # Get Shape of Data
    N, T = data.shape

    # Set number of states to 2, for simplicity
    n_states = 2

    # Array for Predicted Probabilities
    predicted_probabilities = np.zeros([n_states, T + 1])
    
    # Array for Filtered Probabilities
    filtered_probabilities = np.zeros([n_states, T])

    # Array for Log-Likelihood Contributions
    likelihood_contributions = np.zeros(T)

    # Get Transition Probabilities & R matrix from your parameterization function
    p_00, p_11, R = parameterize(random_guesses)
    transition_matrix = create_transition_matrix(p_00, p_11)
    predicted_probabilities[:, 0] = calculate_initial_probabilities(transition_matrix)



    # To Hold values of Forward Filter Recursions
    eta = np.zeros(n_states)
    
    # To Hold values of Forward Filter Recursions
    filters = np.zeros(n_states)
    
    # To Hold values of Partial Log-Likelihoods.
    partial_likelihood = np.zeros(n_states)

    # The Hamilton Filter Loop
    # The Main For Loop:
    for t in range(T):
        # Calculate State Densities
        for state in range(n_states):
            corr_mat = R[state]
            eta[state] = ccc_likelihood_contribution(t, data, corr_mat, standard_deviations)
            partial_likelihood[state] = predicted_probabilities[state,t] * eta[state]
                
      
        #filtering
        filter_0 = eta[0]*predicted_probabilities[0,t]/(eta[0]*predicted_probabilities[0,t]+eta[1]*predicted_probabilities[1,t])
        filter_1 = eta[1]*predicted_probabilities[1,t]/(eta[0]*predicted_probabilities[0,t]+eta[1]*predicted_probabilities[1,t])
        filtered_probabilities[:,t] = filter_0, filter_1     
        
       
        # Calculate the Log-Likelihood
        likelihood_contributions[t] = np.log(np.sum(partial_likelihood[state]))
 
        # Calculate the Prediction step
        predicted_probabilities[[0,1],t+1] = transition_matrix.dot(filtered_probabilities[[0,1],t])
        #predicted_probabilities[:, t+1] = prediction_step(transition_matrix, filtered_probabilities, t)
    
    negative_likelihood = -np.sum(likelihood_contributions)

    return negative_likelihood

def fit(data):
    number_of_correlation_parameters = N * (N - 1) / 2
    
    first_guess = initial_parameters(number_of_correlation_parameters)
    m_bounds = set_bounds(number_of_correlation_parameters)
    print(first_guess)
    standard_deviations = np.zeros((N,T))


    standard_deviations = calculate_standard_deviations(data, univ_params)
    def objective_function(first_guess):
        return Hamilton_Filter(data,first_guess, standard_deviations)
    result = minimize(objective_function, first_guess, bounds=m_bounds, method='L-BFGS-B')
    return result

def plot_heatmaps(result_matrix, labels):
    # Calculate the correlation matrix for the DataFrame
    dims, dimz = result_matrix[0].shape
    print(dims)
    # Set up the matplotlib figure with subplots
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot the Unconditional Correlation heatmap
    sns.heatmap(result_matrix[0], ax=ax[0], annot=True, cmap='coolwarm', xticklabels=labels, yticklabels=labels)
    ax[0].set_title('Conditional Correlation in State 0')
    
    # Plot the Conditional Correlation heatmap
    sns.heatmap(result_matrix[1], ax=ax[1], annot=True, cmap='coolwarm', xticklabels=labels, yticklabels=labels)
    ax[1].set_title('Conditional Correlation in State 1')
    
    # Adjust layout for better appearance
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f'{trie} Hamilton Heatmaps {dims}.png')
    
    # Show the plot
    plt.show()