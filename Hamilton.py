n_states=2
transition_matrix = 0

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
    sigma = params[2:4] ** 2;
    return p_00, p_11, sigma

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
    initial_probs = pi_first,dot(pi_second)
    initial_probabilities = initial_probs.T
    return initial_probabilities

def density_function(t, state, mean, phi sigma):
    return np.exp(-0.5 * np.log(2 * np.pi) - 0.5 * np.log(sigma[state]) - 0.5 * ((data[t] - mean[state] - phi[state] * data[t-1]) ** 2) / sigma[state])


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

def objective(data):
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
    p_00, p_11, sigma = parameterize(params)

    # Form Transition Matrix
    transition_matrix = create_transition_matrix(p_00, p_11)

    # Form Initial Probabilities
    smoothed_probabilities[[0,1],0] = calculate_initial_probabilities(transition_matrix)

    # To Hold values of Forward Filter Recursions
    eta = np.zeros(n_states)

    # To Hold values of Partial Log-Likelihoods.
    partial_likelihood = np.zeros(n_states)

    # The Main For Loop:
    for t in range(num_obs):
        # Calculate State Densities
        for state in range(n_states):
            eta[state] = density_function(t, state, mean=0, phi=0,sigma)
            partial_likelihood[state] = predicted_probabilities[state,t] * eta[state]

        # Calculate the Log-Likelihood

        likelihood_contributions[t] = np.log(np.sum(partial_likelihood))

        # Calculate the Filtering Step
        filtered_probabilities[:,t] = filtering_step(partial_likelihood, likelihood_contributions, t)

        # Calculate the Prediction step
        predicted_probabilities = prediction_step(transition_matrix, filtered_probabilities, t)

    # Return the Negative Sum of the Log-Likelihood
    negative_likelihood = -np.sum(likelihood_contributions)

def fit():
    """
    Minimize the objective Function
        Use initial Guess, constraints, bounds, and arguments.
        Print results.
    """

    # Initial Guesses
    initial_guess= np.array([0.9, 0.9, np.sqrt(2*sigma2), np.sqrt(0.5*sigma2)])

    # Parameter Bounds
    my_bounds = bounds=((0.001,0.9999),(0.001,0.9999),(0.01,None),(0.01,None))

    # Missing: Constraint, args 


    res = minimize(objective, initial_guess, method='L-BFGS-B', bounds=my_bounds) # constraint=cons,arg=args)

    # Hessian, standard errors, print etc
    res.x
    v_hessian = res.hess_inv.todense()
    se_hessian = np.sqrt(np.diagonal(v_hessian))

    # The Results:
    estimated_parameters = res.x
    se = se_hessian
    print('P11='+str(Gamma_hat[0])+', std.errors='+str(se[0]))
    print('P22='+str(Gamma_hat[1])+', std.errors='+str(se[1]))
    print('h1='+str(Gamma_hat[2])+', std.errors='+str(se[2]))
    print('h2='+str(Gamma_hat[3])+', std.errors='+str(se[3]))