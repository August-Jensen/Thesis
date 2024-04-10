import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize


class Base(object):
    """docstring for Base"""
    def __init__(self, dataframe, n_states=2, max_iterations=200, tolerance=1e-6 ): # initial guess,
        # Extract labels and data array from dataframe
        self.dataframe = dataframe
        self.data, self.labels = self.df_to_array(self.dataframe)

        # Set N, T, & number of states
        self.n_states = n_states
        self.N, self.T = self.data.shape

        # Model Settings
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.keep_estimating = True


        # Model Parameters
        self.standard_deviation = np.sqrt(np.var(self.data))
        self.sigma = np.array([0.5 * self.standard_deviation, 2 * self.standard_deviation])
        self.p_00, self.p_11 = 0.98, 0.98
        self.transition_matrix = self.create_transition_matrix(self.p_00, self.p_11) 
        self.initial_state = np.ones(self.n_states) / self.n_states

        # Histories & Convergence Tracking
        self.densities_array = np.zeros((self.n_states,self.T))
        self.forward_probabilities = np.zeros((self.n_states, self.T))
        self.backward_probabilities = np.zeros((self.n_states, self.T))
        
        # Scaling the forward and backward probabilities
        self.scaled_forward_prob = np.zeros((self.n_states, self.T))
        self.scaled_backward_prob = np.zeros((self.n_states, self.T))


        self.filtered_volatility = np.zeros((self.n_states, self.T))
        
        self.smoothed_state_probabilities = np.zeros((self.n_states, self.T))
        self.smoothed_transition_probabilities = np.zeros((self.n_states, self.n_states, self.T))

        # Histories
        self.sigma_histories = np.zeros((self.max_iterations, self.n_states))
        self.markov_histories = np.zeros((self.max_iterations, self.n_states))
        self.log_likelihood_histories = np.zeros(self.max_iterations)
        self.sigma_histories[0, :] = self.sigma 
        self.markov_histories[0,:] = [self.p_00, self.p_11]
        
    def df_to_array(self, dataframe):
        # Create Numpy Array
        data_array = dataframe.to_numpy().T

        # Get Column Labels
        labels = dataframe.columns.tolist()

        return data_array, labels


    def create_transition_matrix(self, p_00, p_11):
        transition_matrix = np.zeros([2,2])

        transition_matrix[0] = p_00, 1 - p_11
        transition_matrix[1] = 1 - p_00, p_11

        transition_matrix = transition_matrix
        # Return the Transition Matrix
        return transition_matrix


    def parameterize(self):
        # Extract diagonals from the transition matrix
        transition_diagonals = np.diag(self.transition_matrix)
        
        # Combine the diagonals and sigma values into a single parameter array
        parameters = np.concatenate((transition_diagonals, self.sigma))
        
        # Prepare bounds: (0.01, 0.99) for transition probabilities, (0.01, None) for sigmas
        transition_bounds = [(0.01, 0.99) for _ in range(self.n_states)]  # Transition probabilities bounds
        sigma_bounds = [(0.01, None) for _ in range(self.n_states)]  # Sigma values bounds
        bounds = transition_bounds + sigma_bounds
        
        return parameters, bounds


    def Density(self, t, sigma):
        """
        The likelihood function within a state at time t
        f(y_t|y_t-1, s_t=j)
        """
        log_liklihood_contribution = - 0.5 * (np.log(2 * np.pi) + np.log(sigma ** 2) + ((self.data[:,t]) ** 2) / (sigma ** 2)) 
        # print(np.exp(log_liklihood_contribution), log_liklihood_contribution)
        return np.exp(log_liklihood_contribution)





    def forward_pass(self,):
        """
        set a_1(j) = 1/n_states f(y_1|s_1=j)
        a_t(j) = sum(f(y_t|s_t=j) * p_ij * a_t-1(i) for i in range(n_states))  
        """
        self.forward_probabilities[:, 0] = self.initial_state * self.densities_array[:,0]
        a_scale = np.sum(self.forward_probabilities[:,0])
        self.scaled_forward_prob[:,0] = self.forward_probabilities[:,0] / a_scale

        forward_array = np.zeros(self.n_states)
        for t in range(1, self.T):
            for j in range(self.n_states):
                forward_array[j] = sum([self.densities_array[j, t] * self.transition_matrix[ i, j] * self.scaled_forward_prob[i, t-1] for i in range(self.n_states)])
                # forward_array[j] = sum([self.densities_array[j, t] * self.transition_matrix[ i, j] * self.forward_probabilities[i, t-1] for i in range(self.n_states)])
            
            a_scale = np.sum(forward_array)
            self.scaled_forward_prob[:,t] = forward_array / a_scale
            #self.forward_probabilities[:,t] = forward_array
        # Return?

    def backward_pass(self,):
        """
        Calculate b_t(i) = sim(f(y_t+1|s_t+1=j)b_t+1(j)p_ij))
        Starting at b_T(j) = 1
        """
        #Set the backwards probabilities at time T
        self.backward_probabilities[:,self.T-1] = np.ones(self.n_states)
        b_scale = np.sum(self.backward_probabilities[:,self.T-1])
        # Looping backwards from T to -1 (to get t= zero.)
        self.scaled_backward_prob[:, self.T-1] = self.backward_probabilities[:, self.T-1] / b_scale

        backward_array = np.zeros(self.n_states)
        for t in range(self.T-2, -1, -1):
            for i in range(self.n_states):
                backward_array[i] = sum([self.densities_array[j,t] * self.scaled_backward_prob[j, t+1] * self.transition_matrix[i,j] for j in range(self.n_states)]) 
                # backward_array[i] = sum([self.densities_array[j,t] * self.backward_probabilities[j, t+1] * self.transition_matrix[i,j] for j in range(self.n_states)]) 
            b_scale = np.sum(backward_array)
            self.scaled_backward_prob[:,t] = backward_array / b_scale
            # self.backward_probabilities[:,t] = backward_array
        # Return?



    def calculate_smoothed_probabilities(self):
        """
        calculate p_t*(j) = (b_t(j) * a_t(j)) / sum(b_t(k) * a_t(k) for k in n_state)
        """
        smoothed_array = np.zeros(self.n_states)
        for t in range(self.T):
            for j in range(self.n_states):
                smoothed_array[j] = self.backward_probabilities[j,t] * self.forward_probabilities[j,t] / sum([self.backward_probabilities[i,t] * self.forward_probabilities[i,t] for i in range(self.n_states)])

            self.smoothed_state_probabilities[:,t] =  smoothed_array




    def calculate_smoothed_transitions(self):
        """
        Calculate p*(i,j) based on forward and backward passes at time t
        for i in range(self.n_states):
            for j in range(self.n_states):

        p_t*(i,j) = (b_t(j) * a_t-1(i) * p_ij * f(y_t|y_t-1, s_t=j) / sum_of_pass
        sum_of_pass = sum(b_t(k) * a_t(k)for k in n_states) 
        """
        smoothed_array = np.zeros((self.n_states,self.n_states))
        for t in range(self.T):
            denominator = sum([self.backward_probabilities[k,t] * self.forward_probabilities[k,t] for k in range(self.n_states)])
            for i in range(self.n_states):
                for j in range(self.n_states):
                    smoothed_array[i,j] = self.backward_probabilities[j,t] * self.forward_probabilities[i,t-1] * self.transition_matrix[i,j]* self.densities_array[j,t] / denominator

            self.smoothed_transition_probabilities[:,:,t] =  smoothed_array



    def scaled_calculate_smoothed_probabilities(self):
        """
        calculate p_t*(j) = (b_t(j) * a_t(j)) / sum(b_t(k) * a_t(k) for k in n_state)
        """
        smoothed_array = np.zeros(self.n_states)
        for t in range(self.T):
            for j in range(self.n_states):
                smoothed_array[j] = self.scaled_backward_prob[j,t] * self.scaled_forward_prob[j,t] / sum([self.scaled_backward_prob[i,t] * self.scaled_forward_prob[i,t] for i in range(self.n_states)])

            self.smoothed_state_probabilities[:,t] =  smoothed_array




    def scaled_calculate_smoothed_transitions(self):
        """
        Calculate p*(i,j) based on forward and backward passes at time t
        for i in range(self.n_states):
            for j in range(self.n_states):

        p_t*(i,j) = (b_t(j) * a_t-1(i) * p_ij * f(y_t|y_t-1, s_t=j) / sum_of_pass
        sum_of_pass = sum(b_t(k) * a_t(k)for k in n_states) 
        """
        smoothed_array = np.zeros((self.n_states,self.n_states))
        for t in range(self.T):
            denominator = sum([self.scaled_backward_prob[k,t] * self.scaled_forward_prob[k,t] for k in range(self.n_states)])
            for i in range(self.n_states):
                for j in range(self.n_states):
                    smoothed_array[i,j] = self.scaled_backward_prob[j,t] * self.scaled_forward_prob[i,t-1] * self.transition_matrix[i,j]* self.densities_array[j,t] / denominator

            self.smoothed_transition_probabilities[:,:,t] =  smoothed_array


    def vectorized_forward_pass(self):
        # Initial step remains the same
        self.forward_probabilities[:, 0] = self.initial_state * self.densities_array[:, 0]
        a_scale = np.sum(self.forward_probabilities[:, 0])
        self.scaled_forward_prob[:, 0] = self.forward_probabilities[:, 0] / a_scale

        # Vectorizing the main loop
        for t in range(1, self.T):
            # Using matrix multiplication to calculate the forward probabilities for all states at once
            forward_array = (self.densities_array[:, t][:, np.newaxis] * 
                             (self.transition_matrix * self.scaled_forward_prob[:, t-1])).sum(axis=0)
            
            # Scaling the probabilities
            a_scale = np.sum(forward_array)
            self.scaled_forward_prob[:, t] = forward_array / a_scale


    def vectorized_backward_pass(self):
        # Set the backward probabilities at time T to 1
        self.backward_probabilities[:, self.T-1] = np.ones(self.n_states)
        b_scale = np.sum(self.backward_probabilities[:, self.T-1])
        self.scaled_backward_prob[:, self.T-1] = self.backward_probabilities[:, self.T-1] / b_scale

        # Vectorizing the loop
        for t in range(self.T-2, -1, -1):
            # Matrix multiplication to compute the backward probabilities
            backward_array = (self.densities_array[:, t+1] * self.scaled_backward_prob[:, t+1]).dot(self.transition_matrix.T)
            
            # Scaling the probabilities
            b_scale = np.sum(backward_array)
            self.scaled_backward_prob[:, t] = backward_array / b_scale


    def vectorized_scaled_calculate_smoothed_probabilities(self):
        # Element-wise multiplication of scaled backward and forward probabilities for each state and time
        product_probabilities = self.scaled_backward_prob * self.scaled_forward_prob
        
        # Summing the products for each time step across all states to get the denominator
        # We use axis=0 to sum across states for each time step
        sum_products = np.sum(product_probabilities, axis=0)
        
        # Broadcasting the sum_products back to the shape of product_probabilities for division
        # This effectively scales each element by the sum of its respective column (time step)
        self.smoothed_state_probabilities = product_probabilities / sum_products

    def vectorized_scaled_calculate_smoothed_transitions(self):
        # Pre-compute the denominator for each time step
        # This is the sum of the element-wise multiplication of scaled_backward_prob and scaled_forward_prob across states
        denominator = np.sum(self.scaled_backward_prob * self.scaled_forward_prob, axis=0)
        
        # For each time step, calculate the smoothed transition probabilities
        for t in range(1, self.T):  # Start from 1 because we need a_t-1
            # Compute the numerator using broadcasting and matrix multiplication
            # The densities_array is aligned with the backward probabilities, and we adjust dimensions for proper broadcasting
            numerator = self.scaled_backward_prob[:, t][:, np.newaxis] * self.densities_array[:, t] * self.transition_matrix * self.scaled_forward_prob[:, t-1]
            
            # Divide by the denominator for each time step, adjusting its shape for broadcasting
            # The denominator needs to be reshaped to ensure the division is broadcasted correctly across the rows
            self.smoothed_transition_probabilities[:, :, t] = numerator / denominator[t]

    def E_Step(self,):
        """
        Run a Forward Pass & a Backward Pass and store results
        Calculate the smoothed_state_probabilities, & the smoothed_transition_probabilities by their respective functions   
        """
        self.forward_pass()
        self.backward_pass()
        self.scaled_calculate_smoothed_transitions()
        self.scaled_calculate_smoothed_probabilities()
        # self.vectorized_forward_pass()
        # self.vectorized_backward_pass()
        # self.vectorized_scaled_calculate_smoothed_transitions()
        # self.vectorized_scaled_calculate_smoothed_probabilities()

    def likelihood_function(self, parameters):
        """
        We maximize the following
        log_likelihood = k + 
                log(p_00) * sum_{t=2}^T(p^*_t(0,0)) 
              + log(1- p_00))* sum_{t=2}^T(p^*_t(0,1)) 
              + log(p_11) * sum_{t=2}^T(p^*_t(1,1)) 
              + log(1- p_11))* sum_{t=2}^T(p^*_t(1,0)) 
              + sum_{t=1}^T(p^*_t(1) * (-0.5 * log(sigma[1] ** 2) - 0.5 * ((x ** 2) / sigma[1] ** 2)))
              + sum_{t=1}^T(p^*_t(2) * (-0.5 * log(sigma[2] ** 2) - 0.5 * ((x ** 2) / sigma[2] ** 2)))
        """
        log_likelihood = 0
        p_0 = parameters[0]
        p_1 = parameters[1]
        sigma = np.zeros(self.n_states)
        sigma[0] = parameters[2]
        sigma[1] = parameters[3]

        transition_guess = self.create_transition_matrix(p_0, p_1)

        for i in range(self.n_states):
            log_likelihood += sum([self.smoothed_state_probabilities[i, t] * np.log(self.Density(t, sigma[i])) for t in range(1, self.T)])
            for j in range(self.n_states):
                log_likelihood += np.log(transition_guess[i,j]) * sum([self.smoothed_transition_probabilities[i,j,t] for t in range(2, self.T)])

        negative_likelihood = -log_likelihood
        #print(f'Negative Log-Likelihood: {negative_likelihood}, Parameters: {parameters}')
        return negative_likelihood

    def M_Step(self):
        """
        Creates a parameter array and bounds.
        Maximize the log-likelihood, by using minimize on the negative log likelihood.
            Update the initial state probabilities
            Update the Transition Matrix
            Update Model Parameters
            Update Log Liklihood
    
        """
        parameters, bounds = self.parameterize()
        def objective_function(parameters):
            return self.likelihood_function(parameters)

        self.result = minimize(objective_function, parameters, method='L-BFGS-B', bounds=bounds, )
        # options={'maxcor': 20, 'ftol': 2.220446049250313e-10, 'gtol': 1e-06, 'maxls': 30})
        # print(self.result.x)
        # print(self.result.success)


    def validate(self, iteration):
        parameter_estimate = self.result.x 
        # print(parameter_estimate)
        
        self.p_00 = parameter_estimate[0]
        self.p_11 = parameter_estimate[1]
        self.sigma[0] = np.sqrt(parameter_estimate[2])
        self.sigma[1] = np.sqrt(parameter_estimate[3])

        self.transition_matrix = self.create_transition_matrix(self.p_00, self.p_11)
        self.sigma_histories[iteration + 1, :] = self.sigma 
        self.markov_histories[iteration + 1, :] = np.array(self.p_00, self.p_11, copy=True)
        # print(type(self.p_00))

    def fit(self,):
        """
        Loops over iteration in max_iterations, and runs the following
            E Step:     Determines the smoothed probabilities and the smoothed transitions
            M Step:     maximizes the parameters based on the probabilities in the E Step using Scipy minimize
            validate:   Updates Histories and Checks Convergence. Later it should 

        """

        # To Break the estimation

        for iteration in range(self.max_iterations):
            for t in range(self.T):
                densities = np.zeros(self.n_states)
                for state in range(self.n_states):
                    #print(self.Density(state, self.data[:,t]))
                    densities[state] = self.Density(t, self.sigma[state])
                self.densities_array[:,t] = densities
            # print(f'Iteration Number: {iteration}')
            self.E_Step()
            self.M_Step()
            self.log_likelihood_histories[iteration] = - self.result.fun
            self.validate(iteration)



            if np.abs(self.log_likelihood_histories[iteration] - self.log_likelihood_histories[iteration-1]) < self.tolerance:
                break
            if iteration == self.max_iterations-2:
                break
    # def test(self,):
    #     """
    #     Loops over iteration in max_iterations, and runs the following
    #         E Step:     Determines the smoothed probabilities and the smoothed transitions
    #         M Step:     maximizes the parameters based on the probabilities in the E Step using Scipy minimize
    #         validate:   Updates Histories and Checks Convergence. Later it should 

    #     """
    #     for t in range(self.T):
    #         densities = np.zeros(self.n_states)
    #         for state in range(self.n_states):
    #             #print(self.Density(state, self.data[:,t]))
    #             densities[state] = self.Density(self.sigma[state], self.data[:,t])
    #         self.densities_array[:,t] = densities
    #     self.forward_pass()
    #     # To Break the estimation



    def summarize(self,):
        """
        
        """
        pass


    def plot_convergence(self,):
        """
        
        """
        pass



    def plot_smoothed_states(self,):
        """
        
        """
        pass


    def plot_volatility(self,):
        """
        
        """
        pass
















class AREM(Base):
    """Extended version of Base class with additional parameters mu and phi"""
    def __init__(self, dataframe, n_states=2, max_iterations=200, tolerance=1e-6):
        # Initialize Base parameters
        super().__init__(dataframe, n_states, max_iterations, tolerance)
        
        # Model Parameters
        self.mu = np.zeros(self.n_states)  # Mean parameter for each state
        self.phi = np.zeros(self.n_states)  # Autoregressive parameter for each state
        
        # Histories
        self.mu_histories = np.zeros((self.max_iterations, self.n_states))  # Tracks mu values across iterations
        self.phi_histories = np.zeros((self.max_iterations, self.n_states))  # Tracks phi values across iterations
        
        # Initialize histories for mu and phi with initial values
        self.mu_histories[0, :] = self.mu
        self.phi_histories[0, :] = self.phi

    def parameterize(self):
        """
        Extends parameterization to include mu and phi for each state.
        """
        # Original parameterization
        parameters, bounds = super(AR, self).parameterize()
        
        # Append mu and phi parameters and their bounds
        mu_phi_parameters = np.concatenate((self.mu, self.phi))
        mu_phi_bounds = [(None, None) for _ in range(2 * self.n_states)]  # No specific bounds for mu and phi
        
        # Combine parameters and bounds
        extended_parameters = np.concatenate((parameters, mu_phi_parameters))
        extended_bounds = bounds + mu_phi_bounds
        
        return extended_parameters, extended_bounds


        
    def parameterize(self):
        # Extract diagonals from the transition matrix
        transition_diagonals = np.diag(self.transition_matrix)
        
        # Combine the diagonals and sigma values into a single parameter array
        parameters = np.concatenate((transition_diagonals, self.mu, self.phi, self.sigma))
        
        # Prepare bounds: (0.01, 0.99) for transition probabilities, (0.01, None) for sigmas
        transition_bounds = [(0.01, 0.99) for _ in range(self.n_states)]  # Transition probabilities bounds
        ar_bounds = [(-0.999, 0.999) for _ in range(2 * self.n_states)]  # Sigma values bounds
        sigma_bounds = [(0.01, None) for _ in range(self.n_states)]  # Sigma values bounds
        bounds = transition_bounds + ar_bounds + sigma_bounds
        
        return parameters, bounds

    def Density(self, t,mu, phi, sigma):
        """
        The likelihood function within a state at time t
        f(y_t|y_t-1, s_t=j) = mu + phi * x_{t-1} + z_t
        """
        log_liklihood_contribution = - 0.5 * (np.log(2 * np.pi) + np.log(sigma ** 2) + ((self.data[:,t] - mu - phi * self.data[:,t-1]) ** 2) / (sigma ** 2)) 
        # print(np.exp(log_liklihood_contribution), log_liklihood_contribution)
        return np.exp(log_liklihood_contribution)



    def likelihood_function(self, parameters):
        """
        We maximize the following
        log_likelihood = k + 
                log(p_00) * sum_{t=2}^T(p^*_t(0,0)) 
              + log(1- p_00))* sum_{t=2}^T(p^*_t(0,1)) 
              + log(p_11) * sum_{t=2}^T(p^*_t(1,1)) 
              + log(1- p_11))* sum_{t=2}^T(p^*_t(1,0)) 
              + sum_{t=1}^T(p^*_t(1) * (-0.5 * log(sigma[1] ** 2) - 0.5 * ((x ** 2) / sigma[1] ** 2)))
              + sum_{t=1}^T(p^*_t(2) * (-0.5 * log(sigma[2] ** 2) - 0.5 * ((x ** 2) / sigma[2] ** 2)))
        """
        log_likelihood = 0
        p_0 = parameters[0]
        p_1 = parameters[1]
        mu = np.zeros(self.n_states)
        mu[0] = parameters[2]
        mu[1] = parameters[3]
        phi = np.zeros(self.n_states)
        phi[0] = parameters[4]
        phi[1] = parameters[5]
        sigma = np.zeros(self.n_states)
        sigma[0] = parameters[6]
        sigma[1] = parameters[7]

        transition_guess = self.create_transition_matrix(p_0, p_1)

        for i in range(self.n_states):
            # XXXX Could add a + 1e-8 or the like to the density function in np.log?
            log_likelihood += sum([self.smoothed_state_probabilities[i, t] * np.log(1e-6 + self.Density(t, mu[i], phi[i], sigma[i])) for t in range(1, self.T)]) 
            for j in range(self.n_states):
                log_likelihood += np.log(transition_guess[i,j]) * sum([self.smoothed_transition_probabilities[i,j,t] for t in range(2, self.T)])

        negative_likelihood = -log_likelihood
        #print(f'Negative Log-Likelihood: {negative_likelihood}, Parameters: {parameters}')
        return negative_likelihood

    import numpy as np

    def vectorized_likelihood_function(self, parameters):
        p_0, p_1 = parameters[0], parameters[1]
        mu = parameters[2:4]
        phi = parameters[4:6]
        sigma = parameters[6:8]

        transition_guess = self.create_transition_matrix(p_0, p_1)

        # Compute the density part of the likelihood for each state
        density_log_likelihood = np.sum([
            self.smoothed_state_probabilities[i, 1:self.T] * np.log(1e-6 + self.Density(np.arange(1, self.T), mu[i], phi[i], sigma[i]))
            for i in range(self.n_states)
        ])

        # Compute the transition part of the likelihood
        transition_log_likelihood = np.sum([
            np.log(transition_guess[i, j]) * np.sum(self.smoothed_transition_probabilities[i, j, 2:self.T])
            for i in range(self.n_states) for j in range(self.n_states)
        ])

        log_likelihood = density_log_likelihood + transition_log_likelihood

        negative_likelihood = -log_likelihood
        return negative_likelihood


    def validate(self, iteration):
        parameter_estimate = self.result.x 
        # print(parameter_estimate)
        
        self.p_00 = parameter_estimate[0]
        self.p_11 = parameter_estimate[1]
        self.mu[0] = parameter_estimate[2]
        self.mu[1] = parameter_estimate[3]
        self.phi[0] = parameter_estimate[4]
        self.phi[1] = parameter_estimate[5]
        self.sigma[0] = np.sqrt(parameter_estimate[6])
        self.sigma[1] = np.sqrt(parameter_estimate[7])
        self.transition_matrix = self.create_transition_matrix(self.p_00, self.p_11)
        self.mu_histories[iteration + 1, :] = self.mu
        self.phi_histories[iteration + 1, :] = self.phi 
        self.sigma_histories[iteration + 1, :] = self.sigma 
        self.markov_histories[iteration + 1, :] = np.array(self.p_00, self.p_11, copy=True)
        # print(type(self.p_00))

    def fit(self,):
        """
        Loops over iteration in max_iterations, and runs the following
            E Step:     Determines the smoothed probabilities and the smoothed transitions
            M Step:     maximizes the parameters based on the probabilities in the E Step using Scipy minimize
            validate:   Updates Histories and Checks Convergence. Later it should 

        """

        # To Break the estimation

        for iteration in range(self.max_iterations):
            for t in range(self.T):
                densities = np.zeros(self.n_states)
                for state in range(self.n_states):
                    #print(self.Density(state, self.data[:,t]))
                    densities[state] = self.Density(t, self.mu[state], self.phi[state], self.sigma[state])
                self.densities_array[:,t] = densities
            # print(f'Iteration Number: {iteration}')
            self.E_Step()
            self.M_Step()
            self.log_likelihood_histories[iteration] = - self.result.fun
            self.validate(iteration)
            # print(f'Iteration Number: {iteration}')



            if np.abs(self.log_likelihood_histories[iteration] - self.log_likelihood_histories[iteration-1]) < self.tolerance:
                break
            if iteration == self.max_iterations-2:
                break