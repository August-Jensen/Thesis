# Import
import numpy as np
import pandas as pd 
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
# Model Structure
	# Initialize,
		# Should take self, Data and N_states, Setting tolerance and max iterations by derault
		# A way to Set up Transition Probabilities, and Parameters

	# Setup p_star = np.zeros(T)
	# volStar = np.zeros(T) #Smoothed volatility.
	# logLik = 0 #Value of EM log-likelihood.
	# parVec  = np.zeros([M, 3]) # Vector for storing parameters.
	# likVec  = np.zeros(M)# Vector for storing log likelihoods.

	# Initial Guess of parameters
	# Initial State probabilities (1/num_obs) 
#Main Class
class EM():
	def __init__(self, data, n_states, tolerance=1e-6, max_iterations=100):
		self.data = data
		self.num_obs = len(data)
		self.n_states = n_states

		self.max_iterations = max_iterations
		self.tolerance = tolerance

		self.transition_matrix = create_transition_matrix()
		self.initial_state_probs = #Array of 1/n_states


		# Add dimension for history with max_iterations
			# Self.parameter_history = []
			# self.forward_values = np.zeros([num_obs, n_states])
			# self.backward_values = np.zeros([num_obs, n_states])
			# self.smoothed_state_probabilities = np.zeros([num_obs, n])
			# self.smoothed_transition_probabilities = np.zeros([num_obs - 1, n_states, n_states])
			# self.smoothed_parameters = np.zeros([num_obs, n_states, num_params])
			# self.likelihood = np.zeros(max_iterations)
	# Create Parameters
	# Create Transition Probabilities
	def create_transition_matrix(self, n_states):
		# Create 
		pass
	# Density Function
	def density_function(self,state, t):
		density = np.exp(-data[t] ** 2 / (2 * sigma[state])) / np.sqrt(2*np.pi*sigma[state])
		return density

	def forward_pass(self, data, state):
		# for t in range(num_obs):	
			# for state in n_states:
			# forwad = density * p_ij * forward)
		pass

	def backward_pass(self, data, state):
		pass

	def e_step(self,state,t):

		p_star[t]
	# EM Main Loop
		# For iteration in max_iteration
		# Reset log likelihood
		# For t in num_obs
        	# Densities of the y_t in states
        	# f1 = np.exp(-y[t]**2/(2*sigmaH_sq))/np.sqrt(2*np.pi*sigmaH_sq)#// Density of y_t in state 1.
        	# f2 = np.exp(-y[t]**2/(2*sigmaL_sq))/np.sqrt(2*np.pi*sigmaL_sq)#// Density of y_t in state 1.			
			# pStar[t] = (f1*p)/(f1*p+f2*(1-p)) #// Smoothed state probability for time t.

		#"Maximize step (M-step)" - Updating equations.
	    # sigmaH_sq = np.sum(pStar*y**2)/np.sum(pStar) #// Estimate of sigmaH_sq.
	    # sigmaL_sq = np.sum((1-pStar)*y**2)/np.sum(1-pStar) # Estimate of sigmaL_sq.
	    # p = np.sum(pStar)/T # Estimate of p.
	    # Compute maximized EM log-likelihood value.
	    # for t in range(T):
	        # f1 = np.exp(-y[t]**2/(2*sigmaH_sq))/np.sqrt(2*np.pi*sigmaH_sq)#// Density of y_t in state 1.
	        # f2 = np.exp(-y[t]**2/(2*sigmaL_sq))/np.sqrt(2*np.pi*sigmaL_sq)#// Density of y_t in state 1.
	        # pStar[t] = (f1*p)/(f1*p+f2*(1-p)) #// Smoothed state probability for time t.
	        # logLik = logLik + pStar[t]*(np.log(f1)+np.log(p)) + (1-pStar[t])*(np.log(f2)+np.log(1-p))
	        # volStar[t] =np.sqrt(sigmaH_sq)*pStar[t] + np.sqrt(sigmaL_sq)*(1-pStar[t])
	    #// Save estimates for iteration m.
	    # parVec[m][0] = sigmaH_sq
	    # parVec[m][1] = sigmaL_sq
	    # parVec[m][2] = p
	    # likVec[m] = logLik

	# likVec #iterations

	def m_step(self, state, t):
		pass

	def grid_search():
		# Begin by maximizing running one time optimization of iterated for
			# For iteration in range(0, 1, 0.1)
				# 4 params is 10^4

	def fit(self, data, state):
		for iteration in range(self.max_iteration):
			# Add Progress bar 
			# e_step
			# m_step
			# if np.abs(likelihood[iteration] - likelihood[iteration - 1]) < self.tolerance:
				# break
			# Newton-Raphson maximization of parameter values


		pass 


	def summary():
		# Present the dara
		pass

	def plot_smoothed_probabilities(What to plot):
		pass 

	def plot_histories():
		# 3d plot of parameter progress?
		pass
