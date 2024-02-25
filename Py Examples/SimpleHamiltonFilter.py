import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import norm
from scipy.optimize import minimize




# parameters
self.transition_probabilities = np.zeros((n_states,n_states-1)) # Assuming nstates-1 is rows
# Generate transition probabilities and populate transition_probabilities matrix
self.transition_matrix = np.vstack((transition_probabilities,np.ones([1,n_states])))
self.A = np.vstack(((np.identity(n_states)-self.transitions_matrix),np.ones([1,n_states])))
self.pi_first = np.linalg.inv(A.T.dot(A)).dot(A.T)
self.pi_second = np.vstack((np.zeros([n_states,1]),np.ones([1,1])))
self.initial_probabilities = self.pi_first.dot(self.pi_second)
xi_10[[0,1],0] = initial_probabilities.T
eta=np.zeros(2)







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

class BaseFilter():
    """docstring for BaseFilter
    This version is for a 2 state SV model.
    Missing
        - Use Dictionary or lists? How to pass transition_matrix, and parameter values into the model.
        - Could allow for drawing random_values, or else setting them by default, if they are not provided when calling
        - Distribution of emissions, and MLE. Normal or t distributed with nu degrees of freedom.
        
    """

    def __init__(self, data, n_states,):
        """
        
        """
        # Set basis parameters
        self.data = data
        self.n_states = n_states
        self.num_obs = len(data)
        self.iteration = 0
        self.variance = np.var(self.data)

        # set up parameters, and parameter bounds
        self.sigmas = [self.variance * 0.5, self.variance * 2]



        # print(f'Parameter Bounds:  {self.parameter_bounds}')

        # set up state and transition probabilities.
        self.transition_matrix = np.zeros((n_states, n_states))
        self.initial_state_probs = np.full(self.n_states, 1.0 / self.n_states)

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
        variance = self.sigmas[state] 
        emissions = norm.pdf(self.data[t], mean,variance)
        #print(emissions)
        return  emissions


    def filtering_step(self, t):
        """
        Calculate the emission probabiliteis, 
        Then multiply emission probability with pedicted probability from prior period
        Sum these, 
        return the filtered probabiliy as the numerator divided by the denominator. 
        """

        for state in range(n_states):
            self.emission_probabilities[state, t] = self.emission_probability(state, t)
            self.numerator[state,t] = self.emission_probability[state,t] * self.predicted_probability[state, t]

        denominator =  np.sum(self.numerator[:, t])

        for state in range(n_states):
            self.filtered_probability[state, t] = self.numerator[state, t] / denominator


    def prediction_step(self, t):
        """
        Calculate the predicted probability for t+1 based on the 
        filtered probabilities at time t, and the transition matrix
        Calculate probability of being in state i at time t, given observed data to time t-1 
        P(s_t = i | X_0:t-1 ; theta) = sum_j=1^n_states p_ji P(s_t-1=j | X_0:t-1 ; theta)
        1. Prediction_prob list
        2. for state in range(n_states): column of transition matrix * Probability of last state being 'state', given data to last period.
        """
        # if t == 0:
        #     # For the first step, we use the initial state probabilities,
        #     self.predicted_probability[:,1] = self.initial_state_probs
        # else:
        for state in range(self.n_states):
            # Calculate the predicted probability for each state at t+1
            self.predicted_probability[state, t+1] = np.sum(
                self.transition_matrix[:, state] * self.filtered_probability[:, t]
            )
        print(self.predicted_probability[state, t+1])
        

    def calculate_log_likelihood_contributions(self, t):
        """
        Calculate the log-likelihood contribution at time t.
        """
        x = self.data[t]

        # Compute log-likelihood for each state. 
        log_likelihoods = np.zeros(self.n_states)
        for state in range(self.n_states):
            mean = 0 # self.mu[state]
            variance = self.sigmas[state] 
            log_likelihoods[state] = -0.5 * np.log(2 * np.pi) - 0.5 * np.log(variance) - ((x - mean) ** 2) / (2 * variance)

        # Multiply each state's log-likelihood by its emission probability and sum
        log_likelihood_contribution = np.sum(log_likelihoods * self.emission_probabilities[:, t])
        # print(f'Log-Likelihood Contribution: {log_likelihood_contribution}')
        return log_likelihood_contribution

    def filter_algorithm(self, estimation_parameters): #params)
        """
        Initialize data Arrays.
        Setup Parameters.
        Set first predicted_probability
        Then run a for loop of:
            filtering_step
            prediciton_step
            calculate likelihood.

        """
        n_states = self.n_states  # Assuming n_states is defined in your class

        # Split estimation_parameters into sigmas and transition matrix parameters
        self.sigmas = estimation_parameters[:n_states]
        probability_parameters = estimation_parameters[n_states:]

        # Update the transition matrix
        self.transition_matrix = self.update_transition_matrix(n_states, probability_parameters)
        print(self.transition_matrix)
        # print(f'Sigma Parameters:  {self.sigmas}')
        # print(f'Transition Matrix: \n  {self.transition_matrix}')
        # print(f'Probabilitity parameters, length:{len(self.probability_parameters)}, parameters: {self.probability_parameters}')
        # setup numpy arrays 
        # Initialize Arrays for storing values
        self.predicted_probability = np.zeros([self.n_states, self.num_obs + 1])
        self.filtered_probability = np.zeros([self.n_states,self.num_obs])
        self.smoothed_probabilities = np.zeros([self.n_states,self.num_obs])
        self.likelihood_contributions = np.zeros(self.num_obs)
        self.emission_probabilities = np.zeros([self.n_states, self.num_obs])
        self.numerator = np.zeros([self.n_states, self.num_obs])
        # Set the first predicted_probability to be initial_state_probs
        # This is also done in prediction step and filtering step
        self.predicted_probability[:, 0] = self.initial_state_probs

        # Set up Transition Matrix probabilities.

        #print('Now running for loop in filter_algorithm')
        for t in range(self.num_obs):
            #print(f'Iteration:  {t}')
            # Perform filtering_step at time t
            self.filtering_step(t)

            # Perform the prediction step for the next time point
            self.prediction_step(t)

            # Calculate and store the log likelihood contribution for time t
            self.likelihood_contributions[t] = self.calculate_log_likelihood_contributions(t)
        total_likelihood = np.sum(self.likelihood_contributions)
        print(f'Total log-likelihood:  {total_likelihood}')
        print(f'Iteration: {self.iteration}')
        self.iteration += 1

        return total_likelihood

    def fit(self, ): # initial_guess, bounds):
        """
        set residuals as the minimize function, that minimizes the filter_algorithm, for initial parameters, and bounds.

        """
        # # Create Transition Matrix
        # combined_bounds = self.parameter_bounds + self.probability_bounds
        # print(combined_bounds)

        # estimation_parameters = np.concatenate([self.sigmas, self.probability_parameters])
        # print(estimation_parameters)
        # Combine parameters
        estimation_parameters = np.concatenate([self.sigmas, self.probability_parameters])
        print("Estimation Parameters:", estimation_parameters)
        # Combine bounds
        combined_bounds = self.parameter_bounds + self.probability_bounds
        print("Combined Bounds:", combined_bounds)
        # Ensure the number of bounds matches the number of parameters
        if len(combined_bounds) != len(estimation_parameters):
            raise ValueError("The number of bounds does not match the number of parameters.")

        res = minimize(self.filter_algorithm, estimation_parameters, method='L-BFGS-B', bounds=combined_bounds)
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






        
"""
Straight roof line, with walls zigzagging

Chapel in garden

Narrow passage between buildings leading to a small garden

Short bridge between buildings

Extension on building supported by beams

Building on straigh wall of cliff

Building built into wall of cliff

Houses in square, with another square above and to one side,

Crossing Stairs

Brook running through building. 

A small tower, with 2 bridges, the to one leading to a room unaccessible otherwise. 

Winding staircase on the outside of tower. 

X crossing stairs. <Up to North & West, Down to South & East>

Building around an entrance with large Gate. One to the Ruins, and one up high, with path between 2 buildings

Clock on large wall outside

Tall room, where a main passage becomes a bridge above a lower level

From down by the waterfall, a larger bridge to the islang?

Diagonal paths between buildings, either 1:1, 1:2 or similar

A stone Stairway to a round terrace overlooking the large cave, next to the bridge to greenouses.

"""