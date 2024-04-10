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
    This is the Base Class for the EM, the functions are
        :__init__:                            
        :df_to_array:                       Turn a dataframe into a numpy array and a list of column labels
        :initial_parameters:                                  
        :initial_state_probabilities:                                  
        :create_trasition_matrix:                                  
        ::                                  
        ::                                  
        ::                                  
        ::                                  
        :forwards_pass:                                  
        :backwards_pass:                                  
        :smoothed_probabilities:                                  
        :smoothed_transition:                                  
        :E_Step:                                  
        :M_Step:                                  
        :fit:                                  
        :summarize:                                  
        :plot ...:                                  
        :plot ...:                                  
        :plot ...:                                  

    """

    def __init__(self, dataframe, n_states=2, univariate_parameters = None, max_iterations=200, tolerance=1e-6):
        # Extract dataframe and column names to numpy array.
        self.dataframe = dataframe
        self.data, self.labels = self.df_to_array(self.dataframe)

        self.n_states = n_states
        self.N, self.T = self.data.shape

        self.max_iterations = max_iterations
        self.tolerance = tolerance

        self.simga = np.array([1, 1])
        self.transition_matrix = self.create_transition_matrix
        self.initial_states = np.array([0.5,0.5])
        self.
        # if univariate_parameters is None:
        #     # Create an instance of Univariate with the dataframe
        #     univariate_instance = Univariate(dataframe, n_states)
        #     # Now call univariate_fit on this instance
        #     self.univariate_parameters, self.univariate_statistics = univariate_instance.univariate_fit()
        # else:
        #     self.univariate_parameters = univariate_parameters


        # Create Parameters
            # self.parameters
            # self.transition_matrix
            # self.correlation_matrix


        # Create Parameter Histories
            # self.correlations
            # self.smoothed_probabilities
            # self.other probabilities? - transition
            # self.likelihood
            # self.smoothed_volatility/smoothed_correlation
            # 


    def df_to_array(self, dataframe):
        # Create Numpy Array
        data_array = dataframe.to_numpy().T

        # Get Column Labels
        labels = dataframe.columns.tolist()

        return data_array, labels


    def initial_parameter(self,):
        """
        What

        Parameters:
        - 

        Reurns:
        -
        
        """

        pass


    def create_transition_matrix(self,):
        """
        What

        Parameters:
        - 

        Reurns:
        -
        
        """

        pass


    def calculate_standard_deviations():
        """
        What

        Parameters:
        - 

        Reurns:
        -
        
        """

        pass

    def forward_pass():
        """
        What

        Parameters:
        - 

        Reurns:
        -
        
        """

        pass

    def backward_pass():
        """
        What

        Parameters:
        - 

        Reurns:
        -
        
        """

        pass

    def smoothed_state_probabilities():
        """
        What

        Parameters:
        - 

        Reurns:
        -
        
        """

        pass

    def smoothed_transition_probabilities():
        """
        What

        Parameters:
        - 

        Reurns:
        -
        
        """

        pass

    def E_Step():
        """
        What

        Parameters:
        - 

        Reurns:
        -
        
        """

        pass


    def M_Step():
        """
        Maximize the the expected log_likelihood w.r.t the parameters
        Uses scipy minimize

        Parameters:
            - expected_parameters

        Reurns:
        -

        """
        pass


    def fit(self, ):
        # setup parameters
        # Calculate standard deviations



        # The main for loop of the EM
        # for iteration in range(self.max_iterations):

            # set log_likelihood to 0

            # Loop over data
            # for t in range(self.T):

                # Calculate State Densities
                # for state in range(self.n_states)
                    # Use likelihood density
            # Then find smoothed probability

            # Perform M step
            # estimate each parameter (Numerically?)
            # estimate transition probability p



            # Compute maximized EM log-likelihood value.
            # E STEP?
            #for t in range(self.T):
                # for state in range(self.n_states) 
                    # f1 = np.exp(-y[t]**2/(2*sigmaH_sq))/np.sqrt(2*np.pi*sigmaH_sq)#// Density of y_t in state 1.
              
                # Find smoothed_ pribability again
                # pStar[t] = (f1*p)/(f1*p+f2*(1-p)) #// Smoothed state probability for time t.
                
                # Claculate log likelihood at each t
                # logLik = logLik + pStar[t]*(np.log(f1)+np.log(p)) + (1-pStar[t])*(np.log(f2)+np.log(1-p))

                # Calculate filtered volatility at time t
                # volStar[t] =np.sqrt(sigmaH_sq)*pStar[t] + np.sqrt(sigmaL_sq)*(1-pStar[t])
            

            #// Save estimates for iteration m.
            # parVec[m][0] = sigmaH_sq
            # parVec[m][1] = sigmaL_sq
            # parVec[m][2] = p
            # likVec[m] = logLik


        # Printed estimates and log likelihood value
        # print('sigma_H='+str(parVec[-1][0]))
        # print('sigma_L='+str(parVec[-1][1]))
        # print('p='+str(parVec[-1][2]))
        # print('loglikelihood function: '+str(likVec[-1]))


        # #Compute the switching variable, s_t.
        # ZeroOnes=np.zeros(T) #State variable
        # for t in range(T):
        #     if(pStar[t]>0.5): # If smoothed transition prob. is higher than 50 pct., then set s_t = 1, othereise s_t = 0.
        #         ZeroOnes[t] = 1
        print('I')

    # def plot_smoothed_probabilities(self,):
    #     # Graph simulated observations and hidden states.
    #     # t=np.arange(0,T)
    #     # fig, axs = plt.subplots(3,figsize=(10,10))
    #     # axs[0].plot(t, y, color='b')
    #     # #axs[0].axhline(linewidth=1, color='k')
    #     # axs[0].set_title('log returns $x_t$')

    #     # axs[1].plot(t, pStar)
    #     # axs[1].set_title('Smoothed probabilities, $P(s_t=1|x_t)$')

    #     # axs[2].plot(t, volStar)
    #     # axs[2].set_title('Filtered volatility, $E(\sigma_t|x_t)$')

    #     # fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
    #     # plt.show()
    #     pass


    # def plot_convergence(self,):
    #     #Graph estimation: parameters and logLik.
    #     # m=np.arange(0,M)
    #     # fig, axs = plt.subplots(3,figsize=(10,10))
    #     # axs[0].plot(m, parVec[:,0])
    #     # axs[0].set_title('$\sigma_1^2$", "$\sigma_2^2$')

    #     # axs[1].plot(m, parVec[:,1])
    #     # axs[1].set_title('p')

    #     # axs[2].plot(m, likVec[:])
    #     # axs[2].set_title('$L_{EM}$')

    #     # fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
    #     # plt.show()
    #     pass 