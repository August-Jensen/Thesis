import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tqdm.notebook import tqdm  # Use notebook version for Jupyter notebooks and lab

""" Notes from Fin_A py file

    First it gets data, sets max iterations as M and num_obs to T.
        It Defines the following
            pStar, smoothed probabilities, as np.zeros(T)
            volStar, smoothed Volatility, as np.zeros(T)
            log likelihood to 0
            parVec, parameter vector to np.Zeros([num_obs, 3]) we have 4
            likVec np.zeros(M) vector of likelihood values

        The initial parameters:
            sigma_1 = 2
            sigma_2 = 1
            p = 0.5 initial probability s_t =1

    for iteration (m) in max_iteration M:
        set logLik = 0
        for t in range(T):
            find likelihood for both states
                f1 = np.exp(-y[t]**2/(2*sigmaH_sq))/np.sqrt(2*np.pi*sigmaH_sq)
            find the smoothed probability at time t:
                pStar[t] = (f1*p)/np.sum()
"""








# 2 State SV-Estimation Algorithm
class SVModel():
    """docstring for SV-Model"""
    def __init__(self, data, n_states=2, tolerance=1e-6, max_iterations=100):
        # Basic Setup
        self.data = data
        self.n_states = n_states 
        self.num_obs = len(data)
        self.p_11 = 0.95
        self.p_22 = 0.95
        self.tolerance = tolerance
        self.max_iterations = max_iterations

        # Set initial probabilities as 1/n_states)
        self.initial_state_probabilities = [1 / n_states for _ in range(n_states)]  # Uniform initial state probabilities

        # Create a transition_matrix
        self.transition_matrix = self.create_transition_matrix(n_states)


        # Create initial sigma values 
        self.sigma = [5,15] # self.initial_sigmas(data, n_states)
        print(np.var(self.data))


        # For tracking the histories
        self.num_params = 6 # parameters in transition matrix and in model.
        self.parameter_history = np.zeros((self.max_iterations, self.num_params))
        self.forward_values = np.zeros((self.num_obs, self.n_states))
        self.backward_values = np.zeros((self.num_obs, self.n_states))
        self.smoothed_state_probabilities = np.zeros((self.num_obs, self.n_states))
        self.smoothed_transition_probabilities = np.zeros((self.num_obs-1, self.n_states, self.n_states))


    def create_transition_matrix(self, n_states):
        """ Create an initial transition matrix 
            this matrix should have 0.95 on the diagonal to ensure persistence,
            and off-diagonals should be (1-0.95)/n_states"""
        return np.array([[self.p_11,1 - self.p_11],
                    [1 - self.p_22, self.p_22]])



    def initial_sigmas(self, data, n_states):
        """ Find the variance of the data, then set initial sigma as the standard deviation.
            Based on the number of states, it should then space out the initial values, based on this.
            """
        variance = np.var(data)
        sigma = np.sqrt(variance)  # Convert variance to standard deviation
        
        if n_states == 1:
            return np.array([sigma])
        else:
            # Calculate spacing based on the number of sigmas required
            spacing = sigma * 0.1  # Example spacing factor; adjust as needed
            if n_states % 2 == 0:  # For even number of states
                offsets = np.linspace(-spacing * (n_states // 2), spacing * (n_states // 2), n_states)
            else:  # For odd number of states, ensuring one sigma is exactly at the calculated sigma
                offsets = np.linspace(-spacing * (n_states // 2), spacing * (n_states // 2), n_states)
            
            # Adjust so the middle value (or one of the middle values for even n_states) is exactly sigma
            adjusted_sigmas = sigma + offsets
            print(f'The Variance is: {variance}, and the Adusted Sigmas are {adjusted_sigmas}')
            return adjusted_sigmas

    def flatten_parameters(self):
        """
            """
        # Flatten the transition matrix and sigma into a single array
        flat_transition_matrix = self.transition_matrix.flatten()
        flat_sigmas = np.array(self.sigma)  # Assuming self.sigmas is already a 1D array
        return np.concatenate([flat_sigmas, flat_transition_matrix])


    def update_parameters(self, params):
        """
            """
        # Assuming you know the dimensions of the transition matrix and the length of sigmas
        n_states = self.n_states
        sigma_length = n_states  # Number of sigmas
        transition_matrix_size = n_states * n_states  # Total elements in the transition matrix

        # Extract sigmas and the transition matrix from the params array
        self.sigma = params[:sigma_length]
        flat_transition_matrix = params[sigma_length:sigma_length + transition_matrix_size]
        
        # Reshape the flat transition matrix back to its original shape
        self.transition_matrix = flat_transition_matrix.reshape((n_states, n_states))




    def density(self, state, t, data):
        """
            """
        term1 = -0.5 * np.log(2 * np.pi)
        term2 = -0.5 * np.log(self.sigma[state] ** 2)
        term3 = - (data[t] ** 2) / (2 * self.sigma[state] ** 2)
        return np.exp(term1 + term2 + term3)

    def negative_log_likelihood(self):
        """
            """ 
        # Calculate the negative log-likelihood using smoothed state probabilities and the density function
        nll = 0  # Initialize the negative log-likelihood
        for t in range(self.num_obs):
            for state in range(self.n_states):
                # Calculate log likelihood contribution for each observation at each state,
                # and weight it by the smoothed state probability for that observation and state
                log_likelihood_contribution = self.density(state, t, self.data) #np.exp(self.density(state, t, self.data))
                nll -= self.smoothed_state_probabilities[t, state] * log_likelihood_contribution

        return nll

    def forward_pass(self, data):        
        """
            """
        # Initialize forward values for the first observation
        for i in range(self.n_states):
            self.forward_values[0, i] = self.initial_state_probabilities[i] * self.density(i, 0, self.data) # np.exp(self.density(i, 0, data))
        
        # Compute forward values for subsequent observations
        for t in range(1, self.num_obs):
            for j in range(self.n_states):
                for i in range(self.n_states):
                    self.forward_values[t, j] += self.forward_values[t-1, i] * self.transition_matrix[i, j] * self.density(j, t, self.data) # np.exp(self.density(j, t, data))
        # print(self.forward_values)
        return self.forward_values

    def backward_pass(self, data):
        """
            """        
        # Initialize backward values for the last observation
        self.backward_values[self.num_obs-1, :] = 1  # All states have a backward value of 1 at the last observation
        
        # Compute backward values for previous observations
        for t in range(self.num_obs-2, -1, -1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    self.backward_values[t, i] += self.backward_values[t+1, j] * self.transition_matrix[i, j] * self.density(j, t+1, self.data) # np.exp(self.density(j, t+1, data))
        
        return self.backward_values 


    def e_step(self):
        """
            """    
        # Ensure data, transition_matrix, and sigmas are appropriately initialized and updated.
        
        # Calculate forward and backward values
        self.forward_values = self.forward_pass(self.data)
        # print(f'forward_values {self.forward_values}')
        self.backward_values = self.backward_pass(self.data)
        # print(f'backward_values {self.backward_values}')
        
        # Calculate smoothed state probabilities
        gamma = np.zeros((self.num_obs, self.n_states))
        for t in range(self.num_obs):
            fwd_bwd = self.forward_values[t, :] * self.backward_values[t, :]
            gamma[t, :] = fwd_bwd / np.sum(fwd_bwd)
        # print(f'gamma {gamma}')
        self.smoothed_state_probabilities = gamma
        
        # Calculate smoothed transition probabilities
        xi = np.zeros((self.num_obs - 1, self.n_states, self.n_states))
        for t in range(self.num_obs - 1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    xi[t, i, j] = (self.forward_values[t, i] * self.transition_matrix[i, j] *
                                   self.density(j, t+1, self.data) *# np.exp(self.density(j, t + 1, self.data)) *
                                   self.backward_values[t + 1, j])
            xi[t, :, :] /= np.sum(xi[t, :, :])
        self.smoothed_transition_probabilities = xi
    

    def objective_function(self, params):
        """
            """    
        # Update model parameters
        self.update_parameters(params)
        
        # Calculate and return the negative log-likelihood
        return self.negative_log_likelihood()

    def m_step(self):
        """
            """    
        # Flatten parameters to create the initial guess array
        initial_params = self.flatten_parameters()

        # Bounds on parameters
        bounds = [(0.1, 100), (0.1, 100),  # Bounds for the first two parameters
          (0.001, 0.999),(0.001, 0.999),(0.001, 0.999),(0.001, 0.999)]  # Bounds for the next four parameters

        # Define the objective function for minimization, using negative log-likelihood
        obj_func = lambda params: self.objective_function(params)
        
        # Perform optimization
        result = minimize(obj_func, initial_params, method='Nelder-Mead', bounds=bounds) # 'L-BFGS-B'
        
        # Update model parameters with the results
        self.update_parameters(result.x)

    def fit(self):
        print(self.sigma)

        print(self.transition_matrix)
        flat_dicks = self.flatten_parameters()
        print(flat_dicks)
        """ runs iterations of the algorithm until max_iterations are reached, or the result has converged.
            An iteration should contain the following:
                1. Run the E-Step
                2. Run the M-Step
                3. Save the Parameters to the histories 
                    self.parameter_history
                    self.smoothed_state_probabilities
                    self.smoothed_transition_probabilities 
                4. If |pa[iteration]-\theta[iteration-1]|<tolerance, break
            After running, present results in table?
            """    
        prev_log_likelihood = None
        progress_bar = tqdm(range(self.max_iterations), desc='Fitting Model', leave=True)

        for iteration in progress_bar:
            # E-Step
            self.e_step()

            # M-Step
            self.m_step()

            # Update progress description (optional)
            current_log_likelihood = -self.negative_log_likelihood()
            progress_bar.set_description(f'Iter {iteration}: Log-likelihood = {current_log_likelihood:.4f}')
            progress_bar.refresh()  # to show immediately the update

            # Convergence check
            if prev_log_likelihood is not None and np.abs(current_log_likelihood - prev_log_likelihood) < self.tolerance:
                progress_bar.set_postfix({'status': 'converged'})
                break
            prev_log_likelihood = current_log_likelihood

        progress_bar.close()  # Close the progress bar
        self.present_results(iteration+1)
        

        # prev_log_likelihood = None
        
        # # Iteration loop
        # for iteration in range(self.max_iterations):
        #     # 1. E-Step
        #     self.e_step()
            
        #     # 2. M-Step
        #     self.m_step()
            
        #     # 3. Save parameters and probabilities to history
        #     self.parameter_history[iteration, :] = self.flatten_parameters()
        #     # Note: For saving smoothed probabilities, you may want to implement a different structure,
        #     # as saving them for each iteration might be memory intensive.
            
        #     # 4. Calculate current log-likelihood and check for convergence
        #     current_log_likelihood = -self.negative_log_likelihood()  # Negative since we want to maximize
        #     if prev_log_likelihood is not None and np.abs(current_log_likelihood - prev_log_likelihood) < self.tolerance:
        #         print(f"Convergence reached at iteration {iteration}.")
        #         break
        #     prev_log_likelihood = current_log_likelihood
        
        # # After running, present results
        # self.present_results(iteration+1)

    def present_results(self, final_iteration):
        # Extract final parameters from history
        final_params = self.parameter_history[final_iteration-1, :]
        # Assuming the first `n_states` entries are sigmas and the rest are transition probabilities
        sigmas = final_params[:self.n_states]
        transition_probabilities = final_params[self.n_states:].reshape((self.n_states, self.n_states))
        
        # Create a DataFrame for a neat presentation
        results = pd.DataFrame({
            'Parameter': ['Sigma(s)', 'Transition Matrix'],
            'Value': [list(sigmas), transition_probabilities.tolist()],
            # Placeholder for standard deviations; you would need additional calculations to determine these
            'Std Dev': [np.nan, np.nan]  
        })
        
        print(results)


    def plot_parameters(self):
        plt.figure(figsize=(12, 8))

        # Assuming the first n_states values in parameter_history are sigmas
        # and the rest are transition probabilities
        n_params = self.num_params
        n_states = self.n_states
        for i in range(n_params):
            param_history = self.parameter_history[:, i]
            if i < n_states:
                label = f'Sigma {i+1}'
            else:
                row = (i - n_states) // n_states
                col = (i - n_states) % n_states
                label = f'Transition ({row+1}, {col+1})'
            plt.plot(param_history, label=label)

        plt.xlabel('Iteration')
        plt.ylabel('Parameter Value')
        plt.title('Parameter Convergence')
        plt.legend()
        plt.show()

        
    def plot_smoothed(self):
        num_obs = len(self.data)
        plt.figure(figsize=(12, 8))

        # Plotting the smoothed probabilities for each state
        for state in range(self.n_states):
            plt.plot(self.smoothed_state_probabilities[:, state], label=f'State {state+1}')

        plt.xlabel('Time')
        plt.ylabel('Probability')
        plt.title('Smoothed State Probabilities')
        plt.legend()
        plt.show()
