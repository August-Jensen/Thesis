import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
import seaborn as sns

class ExpMax():

	def __init__(self, data, n_states, tolerance=1e-6, max_iterations=100):
		self.data = data
		self.num_obs = len(data)
		self.n_states = n_states

		self.max_iterations = max_iterations
        self.tolerance = tolerance

        # Parameters
        self.num_params = 3
        # Initialize Histories for Plotting Parameters
        self.mu_history = []
        self.phi_history = []
        self.sigma_history = []

        # Steps
        self.smoothed_probabilities = np.zeros[self.num_obs]
        self.smoothed_volatility = np.zeros[self.num_obs]
        self.log_likelihood_history = np.zeros[self.max_iterations]
        self.parameters


















#Importing packages we need
import numpy as np
from numpy import genfromtxt #This is used to loading a csv-file as a numpy array
import matplotlib.pyplot as plt #pyplot is used to plot the data

#Locate my folder
folder=''
data=np.genfromtxt(folder+'simulated_data.csv', delimiter=',',) #loading in first 4 columns
y = data[1:,1:2]# 100 times log-returns of the S&P 500 index. January 4, 2010 - till end

y = y.T[0,:] #unpacking numpy array
print(y)
T = len(y) #length of time series
M = 200 # Number of iterations.
pStar = np.zeros(T) # Smoothed state probabilities.
volStar = np.zeros(T) #Smoothed volatility.
logLik = 0 #Value of EM log-likelihood.
parVec  = np.zeros([M, 3]) # Vector for storing parameters.
likVec  = np.zeros(M)# Vector for storing log likelihoods.

# Initial parameter values.
sigmaH_sq = 2 #Variance for state 1 (= H, h1^2).
sigmaL_sq = 1 #Variance for state 2	(= L, h2^2).
p = 0.5 #Probability for s_t = 1 (=H).

#EM Estimation.
for m in range(M):
    #Reset logLik from previous iteration.
    logLik = 0
    for t in range(T):
        f1 = np.exp(-y[t]**2/(2*sigmaH_sq))/np.sqrt(2*np.pi*sigmaH_sq)#// Density of y_t in state 1.
        f2 = np.exp(-y[t]**2/(2*sigmaL_sq))/np.sqrt(2*np.pi*sigmaL_sq)#// Density of y_t in state 1.
        pStar[t] = (f1*p)/(f1*p+f2*(1-p)) #// Smoothed state probability for time t.
    #"Maximize step (M-step)" - Updating equations.
    sigmaH_sq = np.sum(pStar*y**2)/np.sum(pStar) #// Estimate of sigmaH_sq.
    sigmaL_sq = np.sum((1-pStar)*y**2)/np.sum(1-pStar) # Estimate of sigmaL_sq.
    p = np.sum(pStar)/T # Estimate of p.
    # Compute maximized EM log-likelihood value.
    for t in range(T):
        f1 = np.exp(-y[t]**2/(2*sigmaH_sq))/np.sqrt(2*np.pi*sigmaH_sq)#// Density of y_t in state 1.
        f2 = np.exp(-y[t]**2/(2*sigmaL_sq))/np.sqrt(2*np.pi*sigmaL_sq)#// Density of y_t in state 1.
        pStar[t] = (f1*p)/(f1*p+f2*(1-p)) #// Smoothed state probability for time t.
        logLik = logLik + pStar[t]*(np.log(f1)+np.log(p)) + (1-pStar[t])*(np.log(f2)+np.log(1-p))
        volStar[t] =np.sqrt(sigmaH_sq)*pStar[t] + np.sqrt(sigmaL_sq)*(1-pStar[t])
    #// Save estimates for iteration m.
    parVec[m][0] = sigmaH_sq
    parVec[m][1] = sigmaL_sq
    parVec[m][2] = p
    likVec[m] = logLik

likVec #iterations

#Printed estimates and log likelihood value
print('sigma_H='+str(parVec[-1][0]))
print('sigma_L='+str(parVec[-1][1]))
print('p='+str(parVec[-1][2]))
print('loglikelihood function: '+str(likVec[-1]))


#Compute the switching variable, s_t.
ZeroOnes=np.zeros(T) #State variable
for t in range(T):
    if(pStar[t]>0.5): # If smoothed transition prob. is higher than 50 pct., then set s_t = 1, othereise s_t = 0.
        ZeroOnes[t] = 1


# Graph simulated observations and hidden states.
t=np.arange(0,T)
fig, axs = plt.subplots(3,figsize=(10,10))
axs[0].plot(t, y, color='b')
#axs[0].axhline(linewidth=1, color='k')
axs[0].set_title('log returns $x_t$')

axs[1].plot(t, pStar)
axs[1].set_title('Smoothed probabilities, $P(s_t=1|x_t)$')

axs[2].plot(t, volStar)
axs[2].set_title('Filtered volatility, $E(\sigma_t|x_t)$')

fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
plt.show()







Honey: Adds a touch of sweetness.
Gochujang (Korean chili paste): For heat and depth.
Soy Sauce: For savory depth.
Rice Wine (like mirin): For a touch of sweetness and acidity.
Pear or Apple, grated: For natural sweetness and tenderizing the meat.
Lime Juice: Adds a zesty tang.
Olive Oil: For richness and to bind the marinade.