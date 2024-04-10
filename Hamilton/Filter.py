import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
# =======================================================
# |                The Base Filter                      |
# =======================================================


class Univariate():
    """docstring for Base"""
    def __init__(self, dataframe, n_states=2):
        # Extract dataframe and column names to numpy array.
        self.data, self.labels = self.df_to_array(dataframe)
        self.n_states = n_states
        self.N, self.T = self.data.shape

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
        #print(self.x.shape)
        #print(sigma.shape)
        # Set the Initial Sigma to be Total Unconditional Variance of data
        sigma[0] = np.sqrt(np.var(x))
        #print(sigma)

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
            self.x = self.data[i,:]




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


class Base:
    """
    This is the Base Class for the Hamilton Filter, the functions are
        :df_to_array:                       Turn a dataframe into a numpy array and a list of column labels
        :form_correlation_matrix:           Create 2 correlation matrix from parameters, 
        :calculate_standard_deviations:     Calculates the D Matrix for estimation
        :create_diagonal_matrix:            Creates the D matrix at time t
        :check_correlation_matrix_is_valid: Checks for diagonal ones, elements in -1,1 and PSD
        :initial_parameters:                Creates the list of initial guesses for the parameters. 0.95 for transitions, randomly chosen correlations  
        :set_bounds:                        Sets bounds on parameters  
        :parameterize:                      sets p_00, p_11, uses form_correlation_matrix and collets to an array
        :create_transition_matrix:          creates the trnasition matrix of p_00, p_11  
        :calculate_initial_probabilities:   Calculate the predicted probabilities at t = 0.  
        :likelihood_contribution:           The Likelihood contribution at time t  
        :Hamilton_Filter:                   The main filter  
        :fit:                               Minimize the negative log likelihood from the Hamilton_Filter          
        :plot_heatmap:                      Plot the heatmap of the correlation matrix in state 0 and 1  
        :smoothing_step:                    Run Hamilton filter with smoothing step, using estimated parameters  
        :plot_probabilities:                Plot predicted, filtered and smoothed probabilities.  
        ::  
        ::  
        ::  
        ::  

    """

    def __init__(self, dataframe, univariate_parameters = None, n_states=2):
        # Extract dataframe and column names to numpy array.
        self.dataframe = dataframe
        self.data, self.labels = self.df_to_array(self.dataframe)

        self.n_states = n_states
        self.N, self.T = self.data.shape

        if univariate_parameters is None:
            # Create an instance of Univariate with the dataframe
            univariate_instance = Univariate(dataframe, n_states)
            # Now call univariate_fit on this instance
            self.univariate_parameters, self.univariate_statistics = univariate_instance.univariate_fit()
        else:
            self.univariate_parameters = univariate_parameters

    def df_to_array(self, dataframe):
        # Create Numpy Array
        data_array = dataframe.to_numpy().T

        # Get Column Labels
        labels = dataframe.columns.tolist()

        return data_array, labels





    # Forms the Correlation Matrix from RSDC_correlation_guess
    def form_correlation_matrix(self, multi_guess):
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
    def calculate_standard_deviations(self):
        # Get Data Dimensions

        # Create Array for Standard Deviations
        standard_deviations = np.zeros((self.T,self.N))

        # Calculate Sigmas for each timeseries
        for i in range(self.N):
            # Unpack Univariate Estimates
            omega, alpha, beta = self.univariate_parameters[i]

            # Create array for Sigma values
            sigma = np.zeros(self.T)

            # Set first observation of Sigma to Sample Variance
            sigma[0] = np.sqrt(np.var(self.data[:, i]))

            # Calculate Sigma[t]
            for t in range(1, self.T):
                sigma[t] = max(omega + alpha * np.abs(self.data[i, t-1]) + beta * sigma[t-1], 1e-6)
            # Save Sigmas to Standard Deviation Array
            standard_deviations[:, i] = sigma

        # Return array of all Standard Deviations
        return standard_deviations


    # Creates a Diagonal Matrix of (N x N), with Standard Deviations on Diagonal, and zeros off the Diagonal
    def create_diagonal_matrix(self, t):
        """
        Creates an N x N diagonal matrix with standard deviations at time t on the diagonal,
        and zeros elsewhere. Here, N is the number of time series.

        :param t: Integer, the time index for which the diagonal matrix is created.
        :param standard_deviations: List of numpy arrays, each array contains the standard deviations over time for a variable.
        :return: Numpy array, an N x N diagonal matrix with the standard deviations at time t on the diagonal.
        """
        # Extract the standard deviations at time t for each series
        std = self.standard_deviations.T
        stds_at_t = np.array(std[:,t])

        # Create a diagonal matrix with these values
        diagonal_matrix = np.diag(stds_at_t)
        
        return diagonal_matrix




    # Check if a Correlation Matrix is PSD, Elements in [-1,1], and symmetric.
    def check_correlation_matrix_is_valid(self, correlation_matrix):
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



       # Calculate the correlation matrix from the data
        corr_matrix = np.corrcoef(data, rowvar=False)
        
        # Extract the off-diagonal elements of the correlation matrix as parameters
        off_diagonal_elements = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        
        # Define the number of correlation parameters as the length of the off-diagonal elements
        number_of_correlation_parameters = len(off_diagonal_elements)
        
        # Initialize the parameters list with the first two elements
        parameters = [0.95, 0.95]
        
        # Use the correlations as uniform_randoms
        uniform_randoms = off_diagonal_elements
        
        # Take half the value of these for random_between
        random_between = uniform_randoms / 2
        
        # Extend the parameters list with uniform_randoms and random_between
        parameters.extend(uniform_randoms)
        parameters.extend(random_between)
        
        return parameters

    def initial_parameters(self, number_of_correlation_parameters):

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

    def set_bounds(self, number_of_correlation_parameters):
        # Ensure the input is treated as an integer
        number_of_correlation_parameters = int(number_of_correlation_parameters)
        
        # First two bounds are (0.01, 0.99)
        bounds = [(0.01, 0.99), (0.01, 0.99)]
        
        # The rest are 2 * number_of_correlation_parameters of bounds (-0.99, 0.99)
        bounds.extend([(-0.9, 0.9) for _ in range(2 * number_of_correlation_parameters)])
        
        return bounds

    def parameterize(self, RSDC_guess):
        # Extract Transition Probabilities
        p_00, p_11 = RSDC_guess[0], RSDC_guess[1]

        # Find where to split the parameters, for the remaining parameters.
        split_index = len(RSDC_guess[2:]) // 2

        # Create the arrays of Parameters for Correlation Matrix 0, 1
        correlation_parameters_0 = RSDC_guess[2: 2 + split_index]
        correlation_parameters_1 = RSDC_guess[2 + split_index:]

        # Form the correlation matrix for each state
        correlation_matrix_0 = self.form_correlation_matrix(correlation_parameters_0)
        correlation_matrix_1 = self.form_correlation_matrix(correlation_parameters_1)

        # Collect into a single array
        correlation_matrix = [correlation_matrix_0, correlation_matrix_1]
        correlation_matrix = np.array(correlation_matrix)

        return p_00, p_11, correlation_matrix


    def create_transition_matrix(self, p_00, p_11):
        transition_matrix = np.zeros([2,2])
        transition_matrix[0] = p_00, 1 - p_11
        transition_matrix[1] = 1 - p_00, p_11

        # Return the Transition Matrix
        return transition_matrix
        



    # Calculate Initial State Probabilities by Transition Matrix
    def calculate_initial_probabilities(self, transition_matrix):
        """
        Determine the best guess of the Initial State Probabilities, from Transition Matrix

        Returns:
        - An array of initial probabilities at time t=0
        
        """
        # Needs Comments and expansion
        A_matrix = np.vstack(((np.identity(2)- self.transition_matrix), np.ones([1,2])))
        pi_first = np.linalg.inv(A_matrix.T.dot(A_matrix)).dot(A_matrix.T)
        pi_second = np.vstack((np.zeros([2,1]), np.ones([1,1])))
        initial_probs = pi_first.dot(pi_second)
        initial_probabilities = initial_probs.T

        return initial_probabilities




    def density(self,t, state):
        # What we need in the terms:
        data = self.data.T

        D = self.create_diagonal_matrix(t)
        # R is defined in Total CCC Likelihood 
        
        # Linear Algebra
        det_D = np.linalg.det(D)
        # if det_D <1e-8:
        #     det_D = 1e-6
        inv_D = np.linalg.inv(D)
        
        # if det_R < 1e-8:
        #     punish = det_R
        #     det_R = 1e-6
        # else:
        #     punish = 0
        # inv_R = np.linalg.inv(R)
        # if det_R >= 0 else np.eye(R.shape[0])  # Fallback for negative det_R
        
        # lambda_penalty = 000
       
        # Penalty for negative det_R
        # penalty = lambda_penalty * min(0, punish)  # This will be non-zero only if det_R is negative
        # The Shock Term
        z = inv_D @ data[t]

        # The Terms of the Log Likelihood Contribution
        term_1 = self.N * np.log(2 * np.pi)
        term_2 = 2 * np.log(det_D) 
        term_3 = np.log(self.det_R[state])
        term_4 = z.T @ self.inv_R[state] @ z

        log_likelihood_contribution = -0.5 * (term_1 + term_2 + term_3 + term_4) # -  self.penalty[state]
        
        return np.exp(log_likelihood_contribution)

    def Hamilton_Filter(self, random_guesses):
        # Set number of states to 2, for simplicity

        # Array for Predicted Probabilities
        self.predicted_probabilities = np.zeros([self.n_states, self.T + 1])
        
        # Array for Filtered Probabilities
        self.filtered_probabilities = np.zeros([self.n_states, self.T])

        # Array for Log-Likelihood Contributions
        likelihood_contributions = np.zeros(self.T)

        # Get Transition Probabilities & R matrix from your parameterization function
        p_00, p_11, self.R = self.parameterize(random_guesses)


        # Initialize arrays or lists to hold the determinants and inverses
        self.det_R = np.zeros(2)
        self.inv_R = []

        for i in range(2):
            self.det_R[i] = np.linalg.det(self.R[i])  # Compute determinant of R in each state
            self.inv_R.append(np.linalg.inv(self.R[i]))  # Compute inverse of R in each state

        self.transition_matrix = self.create_transition_matrix(p_00, p_11).T
        self.predicted_probabilities[:, 0] = self.calculate_initial_probabilities(self.transition_matrix)

        # print('The correlation of R in state 0')
        # print(self.R[0])
        print(f'The det_R in 0 {self.det_R[0]} \nThe det_R in 1 {self.det_R[1]}')
        # print(self.det_R[0])
        # # print('The inverse of R in state 0')
        # # print(self.inv_R[0])
        # # print('The correlation of R in state 1')
        # # print(self.R[1])
        # print('The Determinant of R in state 1')
        # print(self.det_R[1])
        # self.penalty = np.zeros(2)
        if self.det_R[1] < 1e-6:
            self.det_R[1] = 100000
            # self.penalty[1] = 1e5
        if self.det_R[0] < 1e-6:
            self.det_R[0] = 100000
            # self.penalty[0] = 1e5
        # else:
            # self.penalty = 0,0
        # print('The inverse of R in state 1')
        # print(self.inv_R[1])
        # To Hold values of Forward Filter Recursions
        eta = np.zeros(self.n_states)
        
        # To Hold values of Forward Filter Recursions
        filters = np.zeros(self.n_states)
        
        # To Hold values of Partial Log-Likelihoods.
        partial_likelihood = np.zeros(self.n_states)

        # The Hamilton Filter Loop
        # The Main For Loop:
        for t in range(self.T):
            # Calculate State Densities
            for state in range(self.n_states):
                eta[state] = self.density(t, state)
                partial_likelihood[state] = self.predicted_probabilities[state,t] * eta[state]
                    
          
            #filtering
            filter_0 = eta[0]*self.predicted_probabilities[0,t]/(eta[0]*self.predicted_probabilities[0,t]+eta[1]*self.predicted_probabilities[1,t])
            filter_1 = eta[1]*self.predicted_probabilities[1,t]/(eta[0]*self.predicted_probabilities[0,t]+eta[1]*self.predicted_probabilities[1,t])
            self.filtered_probabilities[:,t] = filter_0, filter_1     
            
           
            # Calculate the Log-Likelihood
            likelihood_contributions[t] = np.log(np.sum(pafrtial_likelihood[state]))
     
            # Calculate the Prediction step
            self.predicted_probabilities[[0,1],t+1] = self.transition_matrix.dot(self.filtered_probabilities[[0,1],t])
            #self.predicted_probabilities[:, t+1] = prediction_step(transition_matrix, self.filtered_probabilities, t)
        
        negative_likelihood = -np.sum(likelihood_contributions)
        print(f'Negative Log-Likelihood: {negative_likelihood}')
        return negative_likelihood


    def fit(self):
        number_of_correlation_parameters = self.N * (self.N - 1) / 2
        
        first_guess = self.initial_parameters(number_of_correlation_parameters)
        m_bounds = self.set_bounds(number_of_correlation_parameters)

        self.standard_deviations = np.zeros((self.N,self.T))


        self.standard_deviations = self.calculate_standard_deviations()
        print(f'First Guess: \n{first_guess }')
        print(f'Bounds: \n{m_bounds }')
        print(f'Standard Deviations:\n{self.standard_deviations}')
        #self.p_00, self.p_11, self.R = self.parameterize(first_guess)

        def objective_function(first_guess):
            return self.Hamilton_Filter(first_guess)
        self.result = minimize(objective_function, first_guess, bounds=m_bounds, method='L-BFGS-B')

        return self.result


    # p1, p2, result_matrix = parameterize(fitted.x)
    def plot_heatmaps(self):
        # Calculate the correlation matrix for the DataFrame
        p0, p1, self.result_matrix =  self.parameterize(self.result.x)
        dims, dimz = self.result_matrix[0].shape
        print('Verify that the First Correlation Matrix is Valid!')
        print(self.check_correlation_matrix_is_valid(self.result_matrix[0]))
        
        print('Verify that the Second Correlation Matrix is Valid!')
        print(self.check_correlation_matrix_is_valid(self.result_matrix[1]))
        # Set up the matplotlib figure with subplots
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        
        # Plot the Unconditional Correlation heatmap
        sns.heatmap(self.result_matrix[0], ax=ax[0], annot=True, cmap='coolwarm', xticklabels=self.labels, yticklabels=self.labels)
        ax[0].set_title('Conditional Correlation in State 0')
        
        # Plot the Conditional Correlation heatmap
        sns.heatmap(self.result_matrix[1], ax=ax[1], annot=True, cmap='coolwarm', xticklabels=self.labels, yticklabels=self.labels)
        ax[1].set_title('Conditional Correlation in State 1')
        
        # Adjust layout for better appearance
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(f'Hamilton Heatmap Bitch.png')
        
        # Show the plot
        plt.show()




    def smoothing_step(self):
        # Get Shape of Data


        # Get Number of Correlation Parameters
        number_of_correlation_parameters = self.N * (self.N - 1) / 2

            
        # # Array for Predicted Probabilities
        # self.predicted_probabilities = np.zeros([self.n_states, self.T + 1])
        
        # # Array for Filtered Probabilities
        # self.filtered_probabilities = np.zeros([self.n_states, self.T])

        # # Array for Smoothed Probabilities
        self.smoothed_probabilities = np.zeros([self.n_states, self.T])

        # # Array for Log-Likelihood Contributions
        # likelihood_contributions = np.zeros(self.T)
        
        # # filtered_volatility = np.zeros([N,T])
        # # Get Transition Probabilities & R matrix from your parameterization function
        p_00, p_11, self.o= self.parameterize(self.result.x)
        # self.det_R = np.zeros(2)
        # self.inv_R = []
        # for i in range(2):
        #     self.det_R[i] = np.linalg.det(self.R[i])  # Compute determinant of R in each state
        #     self.inv_R.append(np.linalg.inv(self.R[i]))  # Compute inverse of R in each state
        print('determinant')
        print(self.det_R)

        print('inverted')
        print(self.inv_R)
        #transition_matrix = self.create_transition_matrix(p_00, p_11)

        # self.predicted_probabilities[:, 0] = self.calculate_initial_probabilities(transition_matrix)
        
        # eta = np.zeros(self.n_states)
        
        # # To Hold values of Forward Filter Recursions
        # filters = np.zeros(self.n_states)
        
        # # To Hold values of Partial Log-Likelihoods.
        # partial_likelihood = np.zeros(self.n_states)

        # # The Hamilton Filter Loop
        # for t in range(self.T):
            
        #     # Calculate State Densities
        #     for state in range(self.n_states):
        #         eta[state] = self.density(t, state)
        #         partial_likelihood[state] = self.predicted_probabilities[state,t] * eta[state]
                    
          
        #     #filtering
        #     filter_0 = eta[0]*self.predicted_probabilities[0,t]/(eta[0]*self.predicted_probabilities[0,t]+eta[1]*self.predicted_probabilities[1,t])
        #     filter_1 = eta[1]*self.predicted_probabilities[1,t]/(eta[0]*self.predicted_probabilities[0,t]+eta[1]*self.predicted_probabilities[1,t])
        #     self.filtered_probabilities[:,t] = filter_0, filter_1     
            
           
        #     # Calculate the Log-Likelihood
        #     likelihood_contributions[t] = np.log(np.sum(partial_likelihood[state]))
     
        #     # Calculate the Prediction step
        #     self.predicted_probabilities[[0,1],t+1] = transition_matrix.dot(self.filtered_probabilities[[0,1],t])
        #     #self.predicted_probabilities[:, t+1] = prediction_step(transition_matrix, self.filtered_probabilities, t)
        
        #     # Backwards Smoother
        for t in range(self.T):
            self.smoothed_probabilities[:,self.T-1]=self.filtered_probabilities[:,self.T-1]
            for t in range(self.T-2, 0, -1):
                # print(self.filtered_probabilities[:,t])
                # print(self.smoothed_probabilities[:,t+1])
                # print(self.predicted_probabilities[:,t+1])
                self.smoothed_probabilities[:,t] = self.filtered_probabilities[:,t] * (self.transition_matrix.T.dot(self.smoothed_probabilities[:,t+1] / self.predicted_probabilities[:,t+1]))


            # print(f' Likelihood Value :  {likelihood_contributions[t]}')
            # print(f'  Predicted Probability:  {self.predicted_probabilities[:, t+1]}')
            # print(f' Filtered Probability :  {self.filtered_probabilities[:,t] }')
            # print(f'Eta  :  {eta}')
            # print(f'Partial Likelihood  :  {partial_likelihood}')
        # Find the Total Log Likelihood
        # self.likelihood = np.sum(likelihood_contributions)
        
        # Return the Sum of the Log-Likelihood
        #return self.predicted_probabilities, self.filtered_probabilities, self.smoothed_probabilities, self.likelihood
        return self.smoothed_probabilities





    def plot_my_probabilities(self):

        x = np.arange(self.T)
        fig, ax = plt.subplots(3, figsize=(16, 9))
        fig.subplots_adjust(hspace=0.3)  # Adjust space between plots

        # Using fill_between for all plots to remove lines
        ax[0].fill_between(x, 0, 1 - self.predicted_probabilities[0, :-1], color='darkblue', alpha=0.7)
        ax[0].fill_between(x, 1 - self.predicted_probabilities[0, :-1], 1, color='darkorange', alpha=0.0)
        ax[1].fill_between(x, 0, 1 - self.filtered_probabilities[0, :], color='darkblue', alpha=0.7)
        ax[1].fill_between(x, 1 - self.filtered_probabilities[0, :], 1, color='darkorange', alpha=0.0)
        ax[2].fill_between(x, 0, 1 - self.smoothed_probabilities[0, :], color='darkblue', alpha=0.7)
        ax[2].fill_between(x, 1 - self.smoothed_probabilities[0, :], 1, color='darkorange', alpha=0.0)

        # Setting titles and limits
        titles = ['Predicted state probability, $P(s_t=1|x_{t-1},x_{t-2},...,x_{1})$',
                  'Filtered state probability, $P(s_t=1|x_{t},x_{t-1},...,x_{1})$',
                  'Smoothed state probability, $P(s_t=1|x_{T},x_{T-1},...,x_{1})$']
        for i, axi in enumerate(ax):
            axi.set_xlim(0, self.T)
            axi.set_ylim(0, 1)
            axi.title.set_text(titles[i])
            axi.axhline(0, color='black', linestyle="--")
            axi.axhline(1, color='black', linestyle="--")

        plt.show()

    














class RSDC:
    """
    This is the Base Class for the Hamilton Filter, the functions are
        :form_correlation_matrix:           Create 2 correlation matrix from parameters, collect to an array
        :calculate_standard_deviations:     

    """













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













