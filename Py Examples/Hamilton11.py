n_states=2
print(data)
def parameterize(params):
    """
    takes parameters,
    creates: 
        p_00, p_11
        the sigmas   

    ! This is not expanded to fit n_states!   
    """
    p_00 = params[0]
    p_11 = params[1]
    sigma = params[2:4];
    mu = params[4:6];
    phi = params[6:8];
    
    return p_00, p_11, sigma, mu, phi

def create_transition_matrix(p_00, p_11):
    """
    Create the Transition Matrix 

    ! This is not expanded to fit n_states!
    """
    transition_matrix = np.zeros([2, 2])
    transition_matrix[0] = p_00, 1 - p_11
    transition_matrix[1] = 1 - p_00, p_11
    return transition_matrix

def calculate_initial_probabilities(transition_matrix):
    """
    Use linalg to find the initial probabilities:

    ! This is not expanded to fit n_states!
    """
    A_matrix = np.vstack(((np.identity(2)- transition_matrix), np.ones([1,2])))
    pi_first = np.linalg.inv(A_matrix.T.dot(A_matrix)).dot(A_matrix.T)
    pi_second = np.vstack((np.zeros([2,1]), np.ones([1,1])))
    initial_probs = pi_first.dot(pi_second)
    initial_probabilities = initial_probs.T
    return initial_probabilities

def density_function(t, state,sigma, mu, phi):
    return np.exp(-0.5 * np.log(2 * np.pi) - 0.5 * np.log(sigma[state]) - 0.5 * ((data[t] - mu[state] - phi[state] * data[t-1]) ** 2) / sigma[state])


def prediction_step(transition_matrix, filtered_probabilities, t):
    """
    Use the dot product of the transition_matrix and the filtered_probability
    """  
    predictions = transition_matrix.dot(filtered_probabilities[:,t])
    return predictions

def filtering_step(partial_likelihood, likelihood_contributions, t):
    """
    For State in n_states
        Set filtered_pribabilities for a stateequal to partial_likelihood for the state divided by total likelihood.
    """
    filtered = np.zeros(n_states)
    for state in range(n_states):
        filtered[state] = partial_likelihood[state] / likelihood_contributions[t]
    return filtered

def objective(initial_guess):
    """
    Setup Arrays for Book Keeping
    Create Parameters
    for t in rainge(num_obs):
        ...
    """
    # For Book Keeping
    num_obs = len(data)
    predicted_probabilities = np.zeros([n_states, num_obs+1])
    filtered_probabilities = np.zeros([n_states, num_obs])
    smoothed_probabilities = np.zeros([n_states, num_obs])
    likelihood_contributions = np.zeros(num_obs)


    # Form Model Parameters
    p_00, p_11, sigma, mu, phi = parameterize(initial_guess)

    # Form Transition Matrix
    transition_matrix = create_transition_matrix(p_00, p_11)

    # Form Initial Probabilities
    predicted_probabilities[[0,1],0] = calculate_initial_probabilities(transition_matrix)

    # To Hold values of Forward Filter Recursions
    eta = np.zeros(n_states)
    
    # To Hold values of Forward Filter Recursions
    filters = np.zeros(n_states)
    
    # To Hold values of Partial Log-Likelihoods.
    partial_likelihood = np.zeros(n_states)

    # The Main For Loop:
    for t in range(num_obs):
        # Calculate State Densities
        for state in range(n_states):
            eta[state] = density_function(t, state, sigma, mu, phi)
            partial_likelihood[state] = predicted_probabilities[state,t] * eta[state]

        # Calculate the Log-Likelihood

        likelihood_contributions[t] = np.log(np.sum(partial_likelihood))

        # Calculate the Filtering Step
        num0 = eta[0] * predicted_probabilities[0,t] / (eta[0] * predicted_probabilities[0,t] + eta[1] * predicted_probabilities[1,t])
        num1 = eta[1] * predicted_probabilities[1,t] / (eta[0] * predicted_probabilities[0,t] + eta[1] * predicted_probabilities[1,t])
        filtered_probabilities[[0,1],t] = num0, num1
        # filtered_probabilities[:,t] = filtering_step(partial_likelihood, likelihood_contributions, t)

        # Calculate the Prediction step
        predicted_probabilities[[0,1],t+1] = transition_matrix.dot(filtered_probabilities[[0,1],t])
        #predicted_probabilities[:, t+1] = prediction_step(transition_matrix, filtered_probabilities, t)
        
        
        
        # print(f' Likelihood Value :  {likelihood_contributions[t]}')
        # print(f'  Predicted Probability:  {predicted_probabilities[:, t+1]}')
        # print(f' Filtered Probability :  {filtered_probabilities[:,t] }')
        # print(f'Eta  :  {eta}')
        # print(f'Partial Likelihood  :  {partial_likelihood}')
    # Return the Negative Sum of the Log-Likelihood
    
    negative_likelihood = -np.sum(likelihood_contributions)

    return negative_likelihood
    
def fit(data):
    """
    Minimize the objective Function
        Use initial Guess, constraints, bounds, and arguments.
        Print results.
    """

    # Initial Guesses
    variance = np.var(data)
    initial_guess= np.array([0.9, 0.9, np.sqrt(2*variance), np.sqrt(0.5*variance),0,0,0,0])

    # Parameter Bounds
    my_bounds = bounds=((0.001,0.9999),(0.001,0.9999),(0.01,None),(0.01,None),(None,None),(None,None),(None,None),(None,None))

    # Missing: Constraint, args 


    res = minimize(objective, initial_guess, method='L-BFGS-B', bounds=my_bounds) # constraint=cons,arg=args)

    # Hessian, standard errors, print etc
    res.x
    v_hessian = res.hess_inv.todense()
    se_hessian = np.sqrt(np.diagonal(v_hessian))

    # The Results:
    estimated_parameters = res.x
    se = se_hessian
    print('P11='+str(estimated_parameters[0])+', std.errors='+str(se[0]))
    print('P22='+str(estimated_parameters[1])+', std.errors='+str(se[1]))
    print('h1='+str(estimated_parameters[2])+', std.errors='+str(se[2]))
    print('h2='+str(estimated_parameters[3])+', std.errors='+str(se[3]))
    print('mu1='+str(estimated_parameters[4])+', std.errors='+str(se[4]))
    print('mu2='+str(estimated_parameters[5])+', std.errors='+str(se[5]))
    print('phi='+str(estimated_parameters[6])+', std.errors='+str(se[6]))
    print('phi2='+str(estimated_parameters[7])+', std.errors='+str(se[7]))

    return estimated_parameters

def smoothed(estimates):
    """
        ...
    """
    # For Book Keeping
    num_obs = len(data)
    predicted_probabilities = np.zeros([n_states, num_obs+1])
    filtered_probabilities = np.zeros([n_states, num_obs])
    smoothed_probabilities = np.zeros([n_states, num_obs])
    likelihood_contributions = np.zeros(num_obs)


    # Form Model Parameters
    p_00, p_11, sigma, mu, phi = parameterize(estimates)

    # Form Transition Matrix
    transition_matrix = create_transition_matrix(p_00, p_11)

    # Form Initial Probabilities
    predicted_probabilities[[0,1],0] = calculate_initial_probabilities(transition_matrix)

    # To Hold values of Forward Filter Recursions
    eta = np.zeros(n_states)
    
    # To Hold values of Forward Filter Recursions
    filters = np.zeros(n_states)
    
    # To Hold values of Partial Log-Likelihoods.
    partial_likelihood = np.zeros(n_states)
    filtered_volatility = np.zeros(num_obs)
    # The Main For Loop:
    for t in range(num_obs):
        # Calculate State Densities
        for state in range(n_states):
            eta[state] = density_function(t, state, sigma, mu, phi)
            partial_likelihood[state] = predicted_probabilities[state,t] * eta[state]

        # Calculate the Log-Likelihood

        likelihood_contributions[t] = np.log(np.sum(partial_likelihood))

        # Calculate the Filtering Step
        num0 = eta[0] * predicted_probabilities[0,t] / (eta[0] * predicted_probabilities[0,t] + eta[1] * predicted_probabilities[1,t])
        num1 = eta[1] * predicted_probabilities[1,t] / (eta[0] * predicted_probabilities[0,t] + eta[1] * predicted_probabilities[1,t])
        filtered_probabilities[[0,1],t] = num0, num1
        # filtered_probabilities[:,t] = filtering_step(partial_likelihood, likelihood_contributions, t)

        # Calculate the Prediction step
        predicted_probabilities[[0,1],t+1] = transition_matrix.dot(filtered_probabilities[[0,1],t])
        #predicted_probabilities[:, t+1] = prediction_step(transition_matrix, filtered_probabilities, t)
        

        filtered_volatility[t] = filtered_probabilities[[0],t] * sigma[0] + (1 - filtered_probabilities[[0],t] * sigma[1])
        # Backwards Smoother
        smoothed_probabilities[:,num_obs-1]=filtered_probabilities[:,num_obs-1]
        for t in range(num_obs-2, 0, -1):
            smoothed_probabilities[:,t] = filtered_probabilities[:,t] * (transition_matrix.T.dot(smoothed_probabilities[:,t+1] / predicted_probabilities[:,t+1]))


        # print(f' Likelihood Value :  {likelihood_contributions[t]}')
        # print(f'  Predicted Probability:  {predicted_probabilities[:, t+1]}')
        # print(f' Filtered Probability :  {filtered_probabilities[:,t] }')
        # print(f'Eta  :  {eta}')
        # print(f'Partial Likelihood  :  {partial_likelihood}')
    # Return the Negative Sum of the Log-Likelihood
    return predicted_probabilities, filtered_probabilities, smoothed_probabilities, filtered_volatility


def plot_my_data(data, filtered_volatility):
    fig, ax=plt.subplots(2, figsize=(14,7))
    #fig.set_figheight=(9)
    #fig.set_figwidth=(16)
    fig.suptitle('log-return and filtered volatility')
    ax[0].plot(data,color='r')
    ax[1].plot(np.sqrt(filtered_volatility))

    #Setting titles
    ax[0].title.set_text('Log-returns, $x_t$')
    ax[1].title.set_text('Filtered volatility, $E[\sigma_t|x_t,x_{t-1},...,x_1]$')

def plot_my_probabilities(data, predicted_probabilities, filtered_probabilities, smoothed_probabilities):
    num_obs = len(data)

    #Predicted state probability, Filtered state probability and smoothed state probability
    fig, ax=plt.subplots(3, figsize=(16,9))
    #fig.tight_layout() 

    #Adjusting size between subplots
    fig.subplots_adjust(left=None, bottom=0.025, right=None, top=None, wspace=None, hspace=None)
    #default
    #left  = 0.125  # the left side of the subplots of the figure
    #right = 0.9    # the right side of the subplots of the figure
    #bottom = 0.1   # the bottom of the subplots of the figure
    #top = 0.9      # the top of the subplots of the figure
    #wspace = 0.2   # the amount of width reserved for blank space between subplots
    #hspace = 0.2   # the amount of height reserved for white space between subplots


    ax[0].plot(1 - predicted_probabilities[0,:])
    ax[1].plot(1 - filtered_probabilities[0,:])
    ax[2].plot(1 - smoothed_probabilities[0,:])

    #Setting limits on x axis
    ax[0].set_xlim(0, num_obs)
    ax[1].set_xlim(0, num_obs)
    ax[2].set_xlim(0, num_obs)

    #Setting titles
    ax[0].title.set_text('Predicted state probability, $P(s_t=1|x_{t-1},x_{t-2},...,x_{1})$')
    ax[1].title.set_text('Filtered state probability, $P(s_t=1|x_{t},x_{t-1},...,x_{1})$')
    ax[2].title.set_text('Smoothed state probability, $P(s_t=1|x_{T},x_{T-1},...,x_{1})$')

    #Setting lines at 0 and 1
    ax[0].axhline(0,color='black', linestyle="--")
    ax[0].axhline(1,color='black', linestyle="--")

    ax[1].axhline(0,color='black', linestyle="--")
    ax[1].axhline(1,color='black', linestyle="--")

    ax[2].axhline(0,color='black', linestyle="--")
    ax[2].axhline(1,color='black', linestyle="--")

my_fitted = fit(data)



my_fitted = fit(data)

predicted_probabilities, filtered_probabilities, smoothed_probabilities, filtered_volatility = smoothed(my_fitted)

plot_my_data(data, filtered_volatility)

plot_my_probabilities(predicted_probabilities, filtered_probabilities, smoothed_probabilities)