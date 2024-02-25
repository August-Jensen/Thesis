import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import norm
from scipy.optimize import minimize


"""
When calling, it should be given data and n_states.
    Additionally, it should be possible to specify the following 
        parameters, 
        distribution function,
        generate randomly,
        transition probabilities, 
        initial state probabilits, 
        parameter bounds 
"""
class SVFilter():
    """docstring for BaseFilter
    This version is for a 2 state SV model.
    Missing
        - Use Dictionary or lists? How to pass transition_matrix, and parameter values into the model.
        - Could allow for drawing random_values, or else setting them by default, if they are not provided when calling
        - Distribution of emissions, and MLE. Normal or t distributed with nu degrees of freedom.
        
    """
    def __init__(self, data, n_states,):
        
        # Set basis parameters
        self.data = data
        self.n_states = n_states
        self.num_obs = len(data)

        # set up parameters, and parameter bounds
        self.sigma = self.default_params() 
        
        self.bounds = np.full((n_states, 2), (0.001, None))  # Filling with (0.001, None)
        self.probability_bounds = np.full(((n_states * (n_states -1)), 2), (0.001, 0.999))  # Filling with (0.001, 0.9999)

        # set up state and transition probabilities.
        self.initial_state_probs = self.default_state_probs()
        self.transitions_matrix = initial_transition_matrix if initial_transition_matrix is not None else self.initialize_transition_matrix(n_states)
        
        print(f'Sigma Parameters:  {self.sigma}')
        print(f'Transition Matrix: \n  {self.transitions_matrix}')
        print(f'Initial State Probability:  {self.initial_state_probs}')
        print(f'Parameter Bounds:  {self.bounds}')
        print(f'Bounds on Probabilities:  {self.probability_bounds}')
default params 
default_state_probs
initialize_transition_matrix
    def initialize_transition_matrix(self, n_states):
        """
        Sets a transition matrix with n_states x n_states dimensions.
        Diagonal (probability of staying in a state) has probability 0.95,
        rest are adjusted to sum to 1
        """
        
        # Create a matrix of zeros
        transition_matrix = np.zeros((n_states, n_states))

        # Fill the diagonal with 0.95
        np.fill_diagonal(transition_matrix, 0.95)

        # Calculate the off-diagonal value
        off_diagonal_value = (1 - 0.95) / (n_states - 1)

        # Fill the off-diagonal elements
        for i in range(n_states):
            for j in range(n_states):
                if i != j:
                    transition_matrix[i, j] = off_diagonal_value

        return transition_matrix

    def create_transition_matrix(self,):
        """
        Create a transition matrix for an n_states x n_states Markov model.
        params: A flat list of parameters (length must be n_states*(n_states-1)).
        n_states: The number of states (dimension of the square matrix).
        """
        assert len(self.probability_params) == self.n_states * (self.n_states - 1), "Invalid number of parameters"

        # Reshape the parameters into (n_states-1, n_states) matrix
        param_matrix = np.reshape(self.probability_params, (self.n_states - 1, self.n_states))

        # Create the transition matrix with the last row being 1 minus the sum of the other rows
        transition_matrix = np.vstack([param_matrix, 1 - np.sum(param_matrix, axis=0)])
        
        return transition_matrix
    
    def default_params(self):
        """
        Set the initial parameters to a list if random=False
        Else, set to random
        """
        variance = np.var(self.data)  # Calculate variance of the data
        params = [np.sqrt(variance)]  # Start with the square root of the variance

        for i in range(1, self.n_states):
            params.append(variance ** i)  # Append variance to the power of i

        return params        # if self.random == False:
        #     if self.n_states == 2:
        #         params = [0.5, 2]
        #         return params
        #     elif self.n_states == 3:
        #         params = [0.5, 1, 2]
        #         return params
        #     elif self.n_states == 4:
        #         params = [0.5, 1, 1.5, 2]
        #         return params
        #     elif self.n_states == 5:
        #         params = [0.5, 1, 1.5, 2, 2.5]
        #         return params
        # elif self.random == True:
        #     params = np.random.uniform(0, 5, self.n_states)
        #     return params

    def default_bounds(self, n_states):
        """
        
        """
        # self.bounds
        pass

    def default_state_probs(self):
        initial_state_probs = np.full(self.n_states, 1.0 / self.n_states)
        return initial_state_probs


    def emission_probability(self, state, t):
        """
        Calculate the emission probability for a given state and time t.
        """
        # print(self.mu[state],self.phi[state])
        if t == 0:
            previous_x = 0  # Handle the case for the first observation
        else:
            previous_x = self.data[t-1]

        mean = 0 # self.mu[state] + self.phi[state] * previous_x
        variance = self.sigma[state] ** 2
        emissions = norm.pdf(self.data[t], mean,variance)
        #print(emissions)
        return  emissions


    def filter_step(self, t):
        """
        Calculate the emission probabiliteis, 
        Then multiply emission probability with pedicted probability from prior period
        Sum these, 
        return the filtered probabiliy as the numerator divided by the denominator. 
        """
        emission_probs = np.array([self.emission_probability(state, t) for state in range(self.n_states)])

        # Set predicted probability of prior period
        if t == 0:
            predicted_prob = self.initial_state_probs
        else:
            predicted_prob = self.predicted_probability[:, t-1]

        # Calculate the filtered probability
        numerator = emission_probs * predicted_prob
        denominator = np.sum(numerator)
        filtered_prob = numerator / denominator
        print(filtered_prob)

        # Store results
        self.filtered_probability[:, t] = filtered_prob
        self.emission_probabilities[:, t] = emission_probs


    def prediction_step(self, t):
        """
        Calculate the predicted probability for t+1 based on the 
        filtered probabilities at time t, and the transition matrix

        """
        if t == 0:
            # For the first step, we use the initial state probabilities,
            self.predicted_probability[:,1] = self.initial_state_probs
        else:
            for state in range(self.n_states):
                # Calculate the predicted probability for each state at t+1
                self.predicted_probability[state, t+1] = np.sum(
                    self.transition_matrix[:, state] * self.filtered_probability[:, t]
                )

        

    def calculate_log_likelihood_contributions(self, t):
        """
        Calculate the log-likelihood contribution at time t.
        """
        x = self.data[t]

        # Compute log-likelihood for each state. 
        log_likelihoods = np.zeros(self.n_states)
        for state in range(self.n_states):
            mean = 0 # self.mu[state]
            variance - self.sigma[state] 
            log_likelihoods[state] = -0.5 * np.log(2 * np.pi) - 0.5 * np.log(variance) - ((x - mean) ** 2) / (2 * variance)

        # Multiply each state's log-likelihood by its emission probability and sum
        log_likelihood_contribution = np.sum(log_likelihoods * self.emission_probabilities[:, t])
        return log_likelihood_contribution

    def filter_algorithm(self, ): #params)
        """
        Initialize data Arrays.
        Setup Parameters.
        Set first predicted_probability
        Then run a for loop of:
            filtering_step
            prediciton_step
            calculate likelihood.

        """


        # setup numpy arrays 
        # Initialize Arrays for storing values
        self.predicted_probability = np.zeros([self.n_states, self.num_obs + 1])
        self.filtered_probability = np.zeros([self.n_states,self.num_obs])
        self.smoothed_probabilities = np.zeros([self.n_states,self.num_obs])
        self.likelihood_contributions = np.zeros(self.num_obs)
        self.emission_probabilities = np.zeros([self.n_states, self.num_obs])

        # Set the first predicted_probability to be initial_state_probs
        # This is also done in prediction step and filtering step
        self.predicted_probability[:, 0] = self.initial_state_probs

        # Set up Transition Matrix probabilities.

        print('Now running for loop in filter_algorithm')
        for t in range(self.num_obs):
            print(f'Iteration:  {t}')
            # Perform filtering_step at time t
            self.filtering_step(t)

            # Perform the prediction step for the next time point
            self.prediction_step(t)

            # Calculate and store the log likelihood contribution for time t
            self.likelihood_contributions[t] = self.calculate_log_likelihood_contribution(t)
    

    def fit(self, ): # initial_guess, bounds):
        """
        set residuals as the minimize function, that minimizes the filter_algorithm, for initial parameters, and bounds.

        """
        # Create Transition Matrix
        num_params = self.n_states * (self.n_states - 1)
        self.probability_params = np.random.uniform(0.001, 0.999, size=num_params)
        self.transition_matrix = self.create_transition_matrix()
        print(f'THE NEW PROBABILITY MATRIX:  {self.transition_matrix}')
        predict_parameters = np.array(self.probability_params + self.sigma)

        # Converting bounds to lists
        parameter_bounds_list = self.bounds.tolist()
        probability_bounds_list = self.probability_bounds.tolist()
        combined_bounds = parameter_bounds_list + probability_bounds_list
        print(f'COMBINED BOUNDS:{combined_bounds}')

        print(f'Lendthses. Parameters: {len(predict_parameters)}, Bounds: {len(combined_bounds)}')
        res = minimize(self.filter_algorithm, predict_parameters, method='L-BFGS-B', bounds=combined_bounds)
        print(res)

        Gamma_hat = res.x
        v_hessian = res.hess_inv.todense()
        se_hessian = np.sqrt(np.diagonal(v_hessian))

        # Print results
        for i in range(len(Gamma_hat)):
            print(f'Parameter {i}: {Gamma_hat[i]}, standard error: {se_hessian[i]}')

        return Gamma_hat, se_hessian

    def smoothing_step(self):
        """
        
        """
        
        pass

    def predict(self):
        """
        
        """
        
        pass



    def statistics(self):
        """

        """
        pass

    def plot_results(self):
        """
        
        """
        
        pass

    def plot_smoothed_probabilities(self):
        """
        
        """
        
        pass

    def plot_convergence(self):
        """
        
        """
        
        pass











# import numpy as np
# import pandas as pd
# import statsmodels.api as sm
# import matplotlib.pyplot as plt
# import seaborn as sns
# from statsmodels.graphics.tsaplots import plot_acf
# from scipy.stats import norm
# from scipy.optimize import minimize



# class Hamilton():
#     def __init__(self, n_states, Gamma0, data):
#         """
#         We need to implement the max_iterations, and tolerance
#         """
#         # Setup of Input
#         self.n_states = n_states
#         self.data = data
#         self.num_obs = len(data)

#         # Setup of Probabilities
#         self.transition_matrix = self.initialize_transition_matrix(n_states)
#         self.initial_state_probs = np.full(n_states, 1.0 / n_states)

#         #Setup of Parameters
#         self.mu = [0,0] # (np.random.rand(n_states)* 2)-1  # or set to a specific starting value [-0.4,0,0.4] # 
#         self.phi = [0,0] # (np.random.rand(n_states) * 2)-1 # or set to a specific starting value [-0.1,0,0.1]
#         self.sigma =  Gamma0[3:4]#np.random.rand(n_states)*3   #[0.1,0.1,0.1]
#         print(self.mu, self.phi, self.sigma)

#     def initialize_transition_matrix(self, n_states):
#         """
#         Initializes the transition matrix with specified properties.
#         """
#         # Create an empty matrix
#         if n_states == 2:
#             matrix = np.array([[0.975,0.025],[0.025,0.975]])

#         elif n_states == 3:
#             matrix = np.array([[0.95,0.025,0.025],[0.025,0.95,0.025],[0.025,0.025,0.95]])

#         return matrix

#     def emission_probability(self, state, t):
#         """
#         Calculate the emission probability for a given state and time t.
#         """
#         # print(self.mu[state],self.phi[state])
#         if t == 0:
#             previous_x = 0  # Handle the case for the first observation
#         else:
#             previous_x = self.data[t-1]

#         mean = 0 # self.mu[state] + self.phi[state] * previous_x
#         variance = self.sigma[state] ** 2
#         emissions = norm.pdf(self.data[t], mean,variance)
#         #print(emissions)
#         return  emissions


#     def filtering_step(self, t):
#         # Calculate emission probabilities for each state
#         emission_probs = np.array([self.emission_probability(state, t) for state in range(self.n_states)])

#         if t == 0:
#             predicted_prob = self.initial_state_probs
#         else:
#             predicted_prob = self.predicted_probability[:, t-1]

#         # Calculate filtered probabilities
#         numerator = emission_probs * predicted_prob
#         denominator = np.sum(numerator)
#         filtered_prob = numerator / denominator

#         # Store results
#         self.filtered_probability[:, t] = filtered_prob
#         self.emission_probabilities[:, t] = emission_probs

#     def prediction_step(self, t):
#         """
#         Calculate the predicted probabilities for time t+1 based on the 
#         filtered probabilities at time t and the transition matrix.
#         """
#         if t == 0:
#             # For the first step, use the initial state probabilities
#             self.predicted_probability[:, 1] = self.initial_state_probs
#         else:
#             for state in range(self.n_states):
#                 # Calculate the predicted probability for each state at t+1
#                 self.predicted_probability[state, t+1] = np.sum(
#                     self.transition_matrix[:, state] * self.filtered_probability[:, t]
#                 )

#     def calculate_log_likelihood_contribution(self, t):
#         """
#         Calculate the log likelihood contribution for time t.
#         """
#         x = self.data[t]

#         # Compute log likelihood for each state
#         log_likelihoods = np.zeros(self.n_states)
#         for state in range(self.n_states):
#             mean = self.mu[state]
#             variance = self.sigma[state] ** 2
#             log_likelihoods[state] = -0.5 * np.log(2 * np.pi) - 0.5 * np.log(variance) - ((x - mean) ** 2) / (2 * variance)

#         # Multiply each state's log likelihood by its emission probability and sum
#         log_likelihood_contribution = np.sum(log_likelihoods * self.emission_probabilities[:, t])
#         return log_likelihood_contribution

#     def filter_algorithm(self, params):
#         """
#         Run the filter algorithm and calculate the negative sum of log likelihood contributions.
#         """
#         self.mu, self.phi, self.sigma = params[:self.n_states], params[self.n_states:2*self.n_states], params[2*self.n_states:]

#         # setup numpy arrays 
#         # Initialize Arrays for storing values
#         self.predicted_probability = np.zeros([self.n_states, self.num_obs + 1])
#         self.filtered_probability = np.zeros([self.n_states,self.num_obs])
#         self.smoothed_probabilities = np.zeros([self.n_states,self.num_obs])
#         self.likelihood_contributions = np.zeros(self.num_obs)
#         self.emission_probabilities = np.zeros([self.n_states, self.num_obs])

#         # Set the first predicted_probability to be initial_state_probs
#         self.predicted_probability[:, 0] = self.initial_state_probs

#         for t in range(self.num_obs):
#             # Perform filtering_step at time t
#             self.filtering_step(t)

#             # Perform the prediction step for the next time point
#             self.prediction_step(t)

#             # Calculate and store the log likelihood contribution for time t
#             self.likelihood_contributions[t] = self.calculate_log_likelihood_contribution(t)
    
#     def fit(self, initial_guess, bounds):
#         """
#         Fit the model to the data using scipy.optimize.minimize.
#         """
#         res = minimize(self.filter_algorithm, initial_guess, method='L-BFGS-B', bounds=bounds)
#         Gamma_hat = res.x
#         v_hessian = res.hess_inv.todense()
#         se_hessian = np.sqrt(np.diagonal(v_hessian))

#         # Print results
#         for i in range(len(Gamma_hat)):
#             print(f'Parameter {i}: {Gamma_hat[i]}, standard error: {se_hessian[i]}')

#         return Gamma_hat, se_hessian