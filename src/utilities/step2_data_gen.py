import pandas as pd 
import numpy as np
import igraph
import time 

# this is helper function of generate_CGL_data
def simulate_linear_sem(W, n, sem_type='exp', noise_scale=None):
    """Simulate samples from linear SEM with specified type of noise.
    For uniform, noise z ~ uniform(-a, a), where a = noise_scale.
    Args:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
        n (int): num of samples, n=inf mimics population risk
        sem_type (str): gauss, exp, gumbel, uniform, logistic, poisson
        noise_scale (np.ndarray): scale parameter of additive noise, default all ones
    Returns:
        X (np.ndarray): [n, d] sample matrix, [d, d] if n=inf
    """
    def _simulate_single_equation(X, w, scale):
        """X: [n, num of parents], w: [num of parents], x: [n]"""
        if sem_type == 'gauss':
            z = np.random.normal(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'exp':
            z = np.random.exponential(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'gumbel':
            z = np.random.gumbel(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'uniform':
            z = np.random.uniform(low=-scale, high=scale, size=n)
            x = X @ w + z
        elif sem_type == 'logistic':
            x = np.random.binomial(1, sigmoid(X @ w)) * 1.0
        elif sem_type == 'poisson':
            x = np.random.poisson(np.exp(X @ w)) * 1.0
        else:
            raise ValueError('unknown sem type')
        return x

    d = W.shape[0]
    if noise_scale is None:
        scale_vec = np.ones(d)
    elif np.isscalar(noise_scale):
        scale_vec = noise_scale * np.ones(d)
    else:
        if len(noise_scale) != d:
            raise ValueError('noise scale must be a scalar or has length d')
        scale_vec = noise_scale


    if np.isinf(n):  # population risk for linear gauss SEM
        if sem_type == 'gauss':
            # make 1/d X'X = true cov
            X = np.sqrt(d) * np.diag(scale_vec) @ np.linalg.inv(np.eye(d) - W)
            return X
        else:
            raise ValueError('population risk not available')
    # empirical risk
    G = igraph.Graph.Weighted_Adjacency(W.tolist())
    if not G.is_dag():
        raise ValueError('W must be a DAG')
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    X = np.zeros([n, d])
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=igraph.IN)
        X[:, j] = _simulate_single_equation(X[:, parents], W[parents, j], scale_vec[j])
    return X

# main function: this is main function for generating data for causal graphs discovery (CGL task for LLM)
def generate_CGL_data(n_data = 100, n_var = 4):
    np.random.seed(int(time.time_ns() % 1000000000))
    p_density = np.random.uniform(0.80, 1.0) # set causal graph edge density to a random variable
    edge_weight_low = 2
    edge_weight_high = edge_weight_low + np.random.uniform(2.0, 5.0)
    b_0 = np.random.choice([0, 1], size=(n_var, n_var), p=[1 - p_density, p_density])
    b_0 = (np.tril(b_0, k=-1)).astype(float)
    b_0 = np.transpose(b_0)
    U = np.random.uniform(low=edge_weight_low, high=edge_weight_high, size=[n_var, n_var])
    U[np.random.rand(n_var, n_var) < 0.5] *= -1
    B_0 = (b_0 != 0).astype(float) * U # this is causal graph weighted adjacency matrix
    noise_scale = np.random.uniform(0.5, 1.0) # noise scale of Linear Structural Equation Model, scale of noises
    X_0 = simulate_linear_sem(W=B_0, n=n_data, sem_type='gauss', noise_scale=noise_scale) # by default choose Gaussian noise
    return B_0, X_0 # returns weighted adjacency matrix of the true causal graph and data generated from it (following linear + Gaussian model)

# main function: this is main function for generating HTE or ATE data
def generate_ATE_data(n_data = 1000, n_x_var = 4):
    np.random.seed(int(time.time_ns() % 1000000000))
    # generating X & A (put a while for controlling balance)
    A_data = np.zeros((n_data, 1))
    while not ((n_data / 4 < sum(A_data)) and (sum(A_data) < n_data * 3 / 4)):
        # generating X
        x_means = np.random.uniform(-5.0, 5.0, size = n_x_var)
        x_vars = np.random.uniform(0.5, 1.5, size = n_x_var)
        X_data = np.random.normal(loc=x_means, scale=np.sqrt(x_vars), size=(n_data, n_x_var))
 
        # generating A
        beta_glm_true = np.random.uniform(-5.0, 5.0, size = (n_x_var))# *2))
        logit_elements = np.exp(-np.dot(X_data, beta_glm_true))
        logit = 1 / (1 + logit_elements)
        A_data = np.random.binomial(size = n_data, n=1, p= logit)
        #A_data = np.array([1 if x > 0.5 else 0 for x in logit])
    #print(sum(A_data) / len(A_data))

    # generating Y_true
    concat_entire_data = np.hstack((A_data.reshape(-1, 1), X_data)) #, X_A_inter_data))
    beta_true = np.random.uniform(-5.0, 5.0, size = (1+n_x_var))# *2))
    
    # generating noise
    y_noise_mean = np.random.uniform(-5.0, 5.0)
    y_noise_var = np.random.uniform(0.5, 1.5)
    Y_noise = np.random.normal(loc=y_noise_mean, scale=y_noise_var, size = n_data)
    Y_data = np.dot(concat_entire_data, beta_true) + Y_noise
    #print("true coef:", beta_glm_true, beta_true)
    return beta_true, np.hstack((concat_entire_data, Y_data.reshape(-1, 1)))

def generate_HTE_data(n_data = 1000, n_x_var = 4):
    beta_true, data = generate_ATE_data(n_data = n_data * 5, n_x_var = n_x_var)
    data_small = data[:n_data, :]
    data_large = data[n_data:, :]
    return beta_true, data_small, data_large

# this is main function for generating data for causal graphs discovery (CGL task for LLM)
# returns true_beta, X_data, M_data, Y_data
# true beta is of shape (3,), where true_beta[0] is X-> M, true_beta[1] is X->Y, true_beta[2] is M->Y
# the causal relation is X->M, X->Y, M->Y, X is exposure, M is mediator, Y is final response

def generate_MA_data_prep(n_data = 100):
    np.random.seed(int(time.time_ns() % 1000000000))
    true_beta =  np.random.uniform(low=-5.0, high=5.0, size=5) # true_beta[0] is X-> M, true_beta[1] is X->Y, true_beta[2] is M->Y
    fix_action_one_prob = 0.5
    A_data = np.random.choice([0, 1], size=n_data, p=[fix_action_one_prob, 1-fix_action_one_prob])
    
    x_means = np.random.uniform(-5.0, 5.0, size = 1)
    x_vars = np.random.uniform(0.5, 1.5, size = 1)
    X_data = np.random.normal(loc=x_means, scale=np.sqrt(x_vars), size=n_data) 

    M_noise_mean = np.random.uniform(-1.0, 1.0)
    M_noise_var = np.random.uniform(0.25, 0.75)
    M_noise = np.random.normal(loc=M_noise_mean, scale=M_noise_var, size = n_data)
    M_data = A_data * true_beta[0] + X_data * true_beta[3] + M_noise

    Y_noise_mean = np.random.uniform(-1.0, 1.0)
    Y_noise_var = np.random.uniform(0.25, 0.75)
    Y_noise = np.random.normal(loc=Y_noise_mean, scale=Y_noise_var, size = n_data)
    Y_data = A_data * true_beta[1] + M_data * true_beta[2] + X_data * true_beta[4] +  Y_noise
    #return true_beta, X_data, M_data, Y_data
    return true_beta, np.hstack((A_data.reshape(-1, 1), M_data.reshape(-1, 1), X_data.reshape(-1, 1), Y_data.reshape(-1, 1)))

def generate_MA_data(n_data = 100):
    true_beta, data = generate_MA_data_prep(n_data * 5)
    data_small = data[:n_data, :]
    data_large = data[n_data:, :]
    return true_beta, data_small, data_large

# main function: generate a list of size n_trajectory for n_trajectory trajectories
# each trajectory data is a tuple that contains (action sequence, state sequence, response sequence) in this trajectory
def generate_markov(n_trajectory = 4, n_data = 100, n_x_var = 4, input_true_parameter = None):
    fix_action_one_prob = 0.5
    if input_true_parameter is None:
        #print("input_true_parameter not provided, need to randomly create input_true_parameter as output_true_parameter")
        # specify distribution of initial state (x in first stage, and we assume such distribution is multivar-normal)
        x_init_means = np.random.uniform(-5.0, 5.0, size = n_x_var)
        x_init_vars = np.random.uniform(0.5, 1.5, size = n_x_var)

        # specify true beta of how Y is generated based on X and A, we assume Y = A b_a + X b_x + A * X b_interact
        beta_true = np.random.uniform(-5.0, 5.0, size = (1+n_x_var *2))
        # specify distribution of noise term of y, and we assume this distribution is normal
        y_noise_mean = np.random.uniform(-5.0, 5.0)
        y_noise_var = np.random.uniform(0.5, 1.5)

        # specify transition kernel of state X from stage i to stage i+1, there are two transition kernels, one for action = 0 and the other for action = 1
        # the transition matrix is diagonal matrix, such that even row entries are x_transition_scale * (2 * A - 1); odd row entries are x_transition_scale * (1 - 2 * A)
        x_transition_a_0 = np.zeros((n_x_var, n_x_var))
        x_transition_a_1 = np.zeros((n_x_var, n_x_var))
        # Fill the diagonal with alternating values
        for i in range(n_x_var):
            if i % 2 == 0:
                x_transition_a_0[i, i] = -1
                x_transition_a_1[i, i] = 1
            else:
                x_transition_a_0[i, i] = 1
                x_transition_a_1[i, i] = -1
        x_transition_scale = np.random.uniform(0.0, 1.0)
        x_transition_a_0 = x_transition_a_0 * x_transition_scale
        x_transition_a_1 = x_transition_a_1 * x_transition_scale
        # also specify the distribution of the a noise term for state transition
        x_transition_noise_var = np.random.uniform(0.25, 0.5)

        output_true_parameter = [(x_init_means, x_init_vars), (beta_true, y_noise_mean, y_noise_var), (x_transition_a_0, x_transition_a_1, x_transition_noise_var)]
    else:
        #print("input_true_parameter provided, directly load input_true_parameter as output_true_parameter")
        # input_true_parameter stores  [(x_init_means, x_init_vars), (beta_true, y_noise_mean, y_noise_var), (x_transition_a_0, x_transition_a_1, x_transition_noise_var)]
        x_init_means, x_init_vars = input_true_parameter[0]
        beta_true, y_noise_mean, y_noise_var = input_true_parameter[1]
        x_transition_a_0, x_transition_a_1, x_transition_noise_var = input_true_parameter[2]
        output_true_parameter = input_true_parameter

    # then begin to generate data
    list_trajectory_data = []
    for trajectory_i in range(n_trajectory):
        entire_data = np.zeros([n_data, 1 + n_x_var + n_x_var + 1])

        # first generate initial action, state and response (first row, row 0)
        entire_data[0,0] = np.random.choice([0, 1], size=1, p=[fix_action_one_prob, 1-fix_action_one_prob])
        entire_data[0,1:(1+n_x_var)] = np.random.normal(loc=x_init_means, scale=np.sqrt(x_init_vars), size=n_x_var)
        entire_data[0,(1+n_x_var):(1+n_x_var *2)] = entire_data[0,1:(1+n_x_var)] * entire_data[0,0]
        entire_data[0,-1] = np.dot(entire_data[0,:-1], beta_true) + np.random.normal(loc=y_noise_mean, scale=y_noise_var, size = 1)

        # then generate the rest of rows along this trajectory (row 1 to row n_data-1), following the initial row
        for i in range(1, n_data):
            x_prev = entire_data[i-1,1:(1+n_x_var)]
            a_prev = entire_data[i-1,0]
            if int(a_prev) == 0:
                x_curr = np.dot(x_transition_a_0, x_prev) + np.random.normal(loc=0, scale=np.sqrt(x_transition_noise_var), size=n_x_var)
                # print('transition of x follows 0')
            elif int(a_prev) == 1:
                x_curr = np.dot(x_transition_a_1, x_prev) + np.random.normal(loc=0, scale=np.sqrt(x_transition_noise_var), size=n_x_var)
                # print('transition of x follows 1')
            entire_data[i,0] = np.random.choice([0, 1], size=1, p=[fix_action_one_prob, 1-fix_action_one_prob])
            entire_data[i,1:(1+n_x_var)] = x_curr
            entire_data[i,(1+n_x_var):(1+n_x_var *2)] = x_curr * entire_data[i,0]
            entire_data[i,-1] = np.dot(entire_data[i,:-1], beta_true) + np.random.normal(loc=y_noise_mean, scale=y_noise_var, size = 1)
        A_data = entire_data[:,0]
        X_data = entire_data[:,1:(1+n_x_var)]
        Y_data = entire_data[:,-1]
        #print(A_data.shape, X_data.shape, Y_data.shape)
        # np.hstack((entire_data[:,0].reshape(-1, 1), entire_data[:,1:(1+n_x_var)], entire_data[:,-1].reshape(-1, 1)))
        list_trajectory_data.append((A_data, X_data, Y_data))
    return output_true_parameter, list_trajectory_data

def CPL_return_df(list_trajectory_data):
    A_train = np.vstack([A_data for A_data, _, _ in list_trajectory_data])
    X_train = np.transpose(np.hstack([X_data for _, X_data, _ in list_trajectory_data]))
    Y_train = np.vstack([Y_data for _ , _, Y_data in list_trajectory_data])
    Y_train = np.sum(Y_train, axis = 1).reshape(-1, 1)
    #for x in [A_train, X_train, Y_train]:
    #    print(x.shape)
    return np.hstack([A_train, X_train, Y_train])

def generate_CPL_data(n_trajectory, n_data, n_x_var):
    # generate small data
    np.random.seed(int(time.time_ns()  % 1000000000))
    para, data = generate_markov(n_trajectory, n_data, n_x_var, input_true_parameter = None)
    data_small = CPL_return_df(data)

    # generate large data 
    np.random.seed(int(time.time_ns() % 1000000000))
    para1, data1 = generate_markov(n_trajectory*10, n_data, n_x_var, input_true_parameter = para)
    data_large = CPL_return_df(data1)

    # print para
    assert para == para1

    return data_small, data_large

if __name__ in "__main__":
    data_s, data_l = generate_CPL_data(n_trajectory = 10, n_data = 2, n_x_var = 1)
    #print(data_s.shape, data_l.shape)
    data = pd.DataFrame(data_l)
    data.columns = ["A-0", "A-1", "X-0", "X-1", "Y"]
    data.to_csv("test.csv", index = False)
    #print(data.head())