import numpy as np 
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import norm
import scipy.optimize as opt
# from scipy.optimize import minimize


class HamiltonBase():
    '''
    
    '''
    def __init__(self, data, n_states): # Parameters
        self.num_obs = len(data)
        self.data = data
        self.n_states = n_states
        


    def emission_density(self, t, mean, variance):
        # log_emission = -0.5*np.log(np.pi)-0.5*np.log(variance)-0.5*((t-mean)**2)/variance
        # print(f'Emission Probabilities: {np.exp(log_emission)}')
        return np.exp(-0.5*np.log(np.pi)-0.5*np.log(variance)-0.5*((t-mean)**2)/variance)

    def likelihood(self, parameters):
        p00 = parameters[0]
        p11 = parameters[1]
        sigma = parameters[2:4] ** 2
        # Generate Transition Matrix
        P = np.zeros([self.n_states,self.n_states])
        P[0]=p00, 1-p11
        P[1]=1-p00, p11

        # Bookkeeping
        predicted_prob = np.zeros([self.n_states, self.num_obs+1])
        filtered_prob = np.zeros([self.n_states, self.num_obs])
        smoothed_prob = np.zeros([self.n_states, self.num_obs])
        likelihoods = np.zeros(self.num_obs)

        # Regression:

        A_matrix = np.vstack(((np.identity(self.n_states)-P),np.ones([1,self.n_states])))
        pi_first = np.linalg.inv(A_matrix.T.dot(A_matrix)).dot(A_matrix.T)
        pi_second = np.vstack((np.zeros([self.n_states,1]),np.ones([1,1])))
        pi = pi_first.dot(pi_second)
        # print(f'Pi: {pi.T}')
        predicted_prob[0:self.n_states,0] = pi.T # XXX
        print(predicted_prob[0:self.n_states,0])

        # For loop:
        for t in range(self.num_obs):
            emission_probabilities = np.zeros(self.n_states)
            numerators = np.zeros(self.n_states)
            filtered = np.zeros(self.n_states)
            likelihood_first = np.zeros(self.n_states)
            
            # State Densities
            for state in range(self.n_states):
                #print(f"Sigmas: {sigma[state]}")
                emission_probabilities[state] = self.emission_density(t, 0, sigma[state])
                numerators[state] = emission_probabilities[state] * predicted_prob[state,t]
                #print(f"Emission Probability for state {state} = {emission_probabilities[state]}")
            denominator = np.sum(numerators)

            # Filtering Step
            for state in range(self.n_states):
                filtered[state] = (emission_probabilities[state] * predicted_prob[state,t]) / denominator
                likelihood_first[state] = predicted_prob[state, t] * emission_probabilities[state]
            filtered_prob[0:self.n_states,t] = filtered # XXX

            # Prediction step
            predicted_prob[0:self.n_states, t+1] = P.dot(filtered_prob[0:self.n_states, t])

            # Likelihoods

            print(f"first: {likelihood_first}")
            likelihood_sum = np.sum(likelihood_first)

            print(f"Sum: {likelihood_sum}")
            likelihoods[t] = np.log(likelihood_sum)
            #print(f"Likelihood contribrution: {likelihoods[t]}, Emission Probabilities: {emission_probabilities.round(5)}, predicted_prob: {predicted_prob[0,t]}")
        total_likelihood = -np.sum(likelihoods)
        print(f"Total Likelihood value: {total_likelihood}")
        #print(f"predicted_prob: {predicted_prob}")
        #print(f"Filtered : {filtered_prob}")
        
        return total_likelihood    

    def run_filter(self,):
        variance = np.var(self.data)
        parameters = np.array([0.95,0.95, np.sqrt(2 * variance), np.sqrt(0.5 * variance)]) # Initial Guesses
        #optimizing. We use L-BFGS-B as it allows for bounds and can compute the standard errors (from the inverse hessian) right away
        res = opt.minimize(self.likelihood, parameters, method='L-BFGS-B', bounds=((0.001,0.9999,),(0.001,0.9999,),(0.01,None),(0.01, None)))
        res.x
        
        #retrieves the negative inverse hessian matrix (note we have minimized the negative log likelihood function)
        v_hessian = res.hess_inv.todense()
        se_hessian=np.sqrt(np.diagonal(v_hessian))
        return res, v_hessian, se_hessian

    def fit(self):
        res, v_hessian, se_hessian = self.run_filter()
        estimated_parameters = res.x
        se = se_hessian
        print('P11='+str(estimated_parameters[0])+', std.errors='+str(se[0]))
        print('P22='+str(estimated_parameters[1])+', std.errors='+str(se[1]))
        print('h1='+str(estimated_parameters[2])+', std.errors='+str(se[2]))
        print('h2='+str(estimated_parameters[3])+', std.errors='+str(se[3]))