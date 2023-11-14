# Implementation of: Nejati, A. and Zamani, M., 'Data-Driven Synthesis of Safety Controllers
# via Multiple Control Barrier Certificates', IEEE Control Systems Letters, 2023.


import os
import tomli
import numpy as np
import matplotlib.pyplot as plt
import casadi as cs


from polyhedron_helper import polyhedron2vertices, plot_solid_polyhedra2d, is_inside_polyhedron


def jet_engine_compressor_cs(x_cs, u_cs):
    """
    The compressor dynamics of the jet engine.

    :param x_cs: The state of the compressor.
    :param u_cs: The control input of the compressor.
    :return: The dynamics of the compressor as a casadi function.
    """

    x_1_dot = - x_cs[1] - 3.0 / 2.0 * x_cs[0] ** 2 - 1.0 / 2.0 * x_cs[0] ** 3
    x_2_dot =  x_cs[0] - u_cs
    x_dot = cs.vertcat(x_1_dot, x_2_dot)
    x_dot_func = cs.Function('x_dot_func', [x_cs, u_cs], [x_dot])
    return x_dot_func


def euler_cs(x_cs, u_cs, x_dot_func, dt):
    """
    The euler discretization of the dynamics.

    :param x_cs: The state of the system.
    :param u_cs: The control input of the system.
    :param x_dot_func: The continuous-time dynamics of the system as a casadi function.
    :param dt: The time step.
    :return: The discrete-time dynamics of the system as a casadi function.
    """

    x_next = x_cs + dt * x_dot_func(x_cs, u_cs)
    x_next_func = cs.Function('x_next_func', [x_cs, u_cs], [x_next])
    return x_next_func


# Define the polynomial
def h(x, q_i):
    """
    The polynomial h(x) = q_1 * x_1**4 + q_2 * x_1**2 + q_3 * x_1**2 * x_2**2 + q_4 * x_2**2 + q_5 * x_2**4 + q_6.

    :param x: The state.
    :param q_i: The coefficients of the polynomial.
    :return: The value of the polynomial.
    """

    return q_i[0] * x[0] ** 4 + \
           q_i[1] * x[0] ** 2 + \
           q_i[2] * x[0] ** 2 * x[1] ** 2 + \
           q_i[3] * x[1] ** 2 + \
           q_i[4] * x[1] ** 4 + \
           q_i[5]


def scenario_opt(config, data, X_0, X_u, X_is, num_coeffs=6):
    """
    The scenario optimization problem as defined in the paper.

    :param config: The configuration file.
    :param data: The data set.
    :param X_0: The initial safe set.
    :param X_u: The unsafe set.
    :param X_is: The partioned state space.
    :param num_coeffs: The number of coefficients of the polynomial.
    :return: The solution of the scenario optimization problem.
    """

    # set the configuration parameters
    mu_is_opti = config["mu_is_opti"]
    use_init = config["use_init"]
    has_non_positive_constraint = config["has_non_positive_constraint"]

    N = len(X_is)

    colors = [[[0.5, 0.5, 1.0]], [[1.0, 0.5, 0.5]]]

    # Define the optimization problem
    opti = cs.Opti()

    # Define the decision variables
    eta_i = opti.variable(N, 1)
    eta = opti.variable(1, 1)
    gamma_i = opti.variable(N, 1)
    lambda_i = opti.variable(N, 1)

    if mu_is_opti:
        mu_s = opti.variable(len(data.keys()), 1)  
    else:
        mu_s = 1.0 / len(data.keys()) * np.ones((len(data.keys()), 1))

    q_i = opti.variable(N, num_coeffs)

    cost = eta

    # extract states from data set
    u_0 = list(data.keys())[0]
    tuples = data[u_0]
    x_data = [x for x, _ in tuples]

    if mu_is_opti:
        # Constraints based on Remark 2, after Eq. 11
        mu_sum = 0.0
        for s in range(len(data.keys())):
            mu_sum += mu_s[s]
            opti.subject_to(mu_s[s] >= 0.0)
        opti.subject_to(mu_sum == 1.0)

    for i in range(N):
        X_i = X_is[i]

        opti.subject_to(eta_i[i] <= eta)  # Eq. 9e

        for x in x_data: 
            if is_inside_polyhedron(x, X_i) and is_inside_polyhedron(x, X_0):
                # Add the constraints
                # plt.plot(x[0], x[1], '*', color=colors[i][0])
                opti.subject_to(h(x, q_i[i, :]) - gamma_i[i] <= eta_i[i])  # Eq. 9a
            elif is_inside_polyhedron(x, X_i) and any([is_inside_polyhedron(x, X_u_i) for X_u_i in X_u]):
                # Add the constraints
                # plt.plot(x[0], x[1], '*', color=colors[i][0])
                opti.subject_to(- h(x, q_i[i, :]) + lambda_i[i] <= eta_i[i])  # Eq. 9b
        
        for j in range(N):
            X_j = X_is[j]
            
            opti.subject_to(gamma_i[j] - lambda_i[i] <= eta_i[i])  # Eq. 9c

            for r, x in enumerate(x_data):
                if is_inside_polyhedron(x, X_i):
                    # plt.plot(x[0], x[1], '*', color=colors[i][0])
                    delta_h_sum = 0.0
                    is_constant = True
                    for s in data.keys():
                        x_next = data[s][r][1]
                        if is_inside_polyhedron(x_next, X_j):
                            # plt.plot(x_next[0], x_next[1], '*', color=colors[j][0])
                            is_constant = False
                            delta_h_sum += mu_s[s] * (h(x_next, q_i[j, :]) - h(x, q_i[i, :]) - eta_i[i])  # Eq. 11, part 1
                    if not is_constant:
                        opti.subject_to(delta_h_sum <= 0.0)  # Eq. 11, part 2

    if has_non_positive_constraint:
        opti.subject_to(eta <= 0.0)

    # set the objective
    opti.minimize(cost)

    # warm start the optimization problem
    if use_init:
        if N == 1:
            eta_init = np.array([0.0008])
            q_init = np.array([[0.002, -0.0014, -0.0023, 0.0084, -0.0067, 0.4]])

        elif N == 2:
            eta_init = np.array([-0.0126])
            eta_i_init = np.array([-0.0127, -0.0126])
            gamma_init = np.array([0.5704, 0.5708])
            lambda_init = np.array([0.5830, 0.5812])
            q_init = np.array([[0.002, -0.0025, 0.0037, 0.4, -0.1515, 0.4],
                        [0.002, -0.0318, 0.0507, 0.4, -0.1356, 0.3935]])
            opti.set_initial(eta_i, eta_i_init)
            opti.set_initial(gamma_i, gamma_init)
            opti.set_initial(lambda_i, lambda_init)
        opti.set_initial(eta, eta_init)
        opti.set_initial(q_i, q_init)

    # solve the optimization problem
    p_opts = {"expand": True}
    s_opts = {"max_iter": 1000}
    opti.solver("ipopt", p_opts, s_opts)
    sol = opti.solve()

    # get the solution
    eta_sol = sol.value(eta)
    q_sol = sol.value(q_i)
    gamma_sol = sol.value(gamma_i)
    lambda_sol = sol.value(lambda_i)

    return eta_sol, q_sol, gamma_sol, lambda_sol


def main(config):
    plotting = config["plotting"]
    use_paper_results = config["use_paper_results"]

    # number of CBFs 
    N = config["N"]
    
    # number of data points
    M = config["M"]

    # set the coverage paramter
    eps = config["eps"]

    state_dim = 2
    input_dim = 1

    # Create casadi variables
    x_cs = cs.SX.sym('x_cs', state_dim)
    u_cs = cs.SX.sym('u_cs', input_dim)

    # Create the dynamics model
    f_cont = jet_engine_compressor_cs(x_cs, u_cs)
    delta_t = 0.1
    f_discrete = euler_cs(x_cs, u_cs, f_cont, delta_t)

    # Create discrete control input set
    U = np.linspace(-1.0, 1.0, 21)
    # Create dictionary of control inputs
    U_dict = {}
    for i, u in enumerate(U):
        U_dict[i] = u

    # Create state space
    A = np.array([[1.0, 0.0],
                  [-1.0, 0.0],
                  [0.0, -1.0],
                  [0.0, 1.0], 
                 ])
    b = np.array([1.0, 1.0, 1.0, 1.0])
    X = (A, b)

    # Create initial safe state space
    b = np.array([0.6, 0.6, 0.7, 0.7])
    X_0 = (A, b)

    if N == 1:
        X_is = [X]
    elif N == 2:
        b = np.array([1.0, 0.0, 1.0, 1.0])
        X_i_1 = (A, b)
        b = np.array([0.0, 1.0, 1.0, 1.0])
        X_i_2 = (A, b)
        X_is = [X_i_1, X_i_2]

    # Create unsafe state space
    b = np.array([0.9, 0.9, 1.0, -0.8])
    X_u_1 = (A, b)
    b = np.array([0.9, 0.9, -0.8, 1.0])
    X_u_2 = (A, b)
    X_u = [X_u_1, X_u_2]

    # Create gridded state data in the state space
    x1 = np.linspace(-1.0, 1.0, int(np.sqrt(M)))
    x2 = np.linspace(-1.0, 1.0, int(np.sqrt(M)))
    x_data = np.array(np.meshgrid(x1, x2)).T.reshape(-1, 2)

    eps_data = np.linalg.norm(np.array([x1[0], x2[0]]) - 
                              np.array([x1[1], x2[1]]), 2) / 2.0
    
    # assert eps_data <= eps
    print("epsilon = {}".format(eps_data))

    # Create data dicts of the form {u_key_1: [(x_1, x_1_next), (x_2, x_2_next), ...], u_key_2: [...], ...}
    data = {}
    for u_key in U_dict.keys():
        data[u_key] = []
        u = U_dict[u_key]
        for x in x_data:
            x_next = f_discrete(x, u)
            x_next = np.reshape(np.array(x_next), (state_dim))
            data[u_key].append((x, x_next))
    
    plt.figure(figsize=(20, 20))
    ax = plt.gca()

    if not use_paper_results:
        # Use results from reimplementation
        # Create the optimization problem
        eta, q, gamma, lambda_ = scenario_opt(config, data, X_0, X_u, X_is)
    else:
        assert N == 2  # Cannot plot the results for N = 1 as gamma and lambda are not given in the paper

        # Use results from paper
        eta = - 0.0126
        gamma = np.array([0.5704, 0.5708])
        lambda_ = np.array([0.5830, 0.5812])
        q = np.array([[0.002, -0.0025, 0.0037, 0.4, -0.1515, 0.4],
                      [0.002, -0.0318, 0.0507, 0.4, -0.1356, 0.3935]])
    print("eta = {}".format(eta))
    print("gamma = {}".format(gamma))
    print("lambda = {}".format(lambda_))
    print("q = {}".format(q))

    if plotting:
        # create a directory for the figures if the don't exist yet
        if not os.path.exists('figures'):
            os.makedirs('figures')

        plot_solid_polyhedra2d(X, color='k', alpha=0.2)
        plot_solid_polyhedra2d(X_0, color='g', alpha=0.5)
        for X_u_i in X_u:
            plot_solid_polyhedra2d(X_u_i, color='r', alpha=0.5)

        colors = [[0.2, 0.2, 1.0], [1.0, 0.2, 0.2]]
        linestyles = ['solid', 'dashed']

        labels = []
        handles = []

        for j in range(N):
            x_vertices, y_vertices = polyhedron2vertices(X_is[j])
            x1 = np.linspace(min(x_vertices), max(x_vertices), 100)
            x2 = np.linspace(min(y_vertices), max(y_vertices), 100)
            X1, X2 = np.meshgrid(x1, x2)
            X = np.vstack((X1.flatten(), X2.flatten()))

            if N == 1:
                q_j = q
                gamma_j = gamma
                lambda_j = lambda_
                colors_j = colors[0]
            else:
                q_j = q[j, :]
                gamma_j = gamma[j]
                lambda_j = lambda_[j]
                colors_j = colors[j]

            Y = np.zeros(X.shape[1])
            for i in range(X.shape[1]):
                Y[i] = h(X[:, i], q_j)
            Y = Y.reshape(X1.shape)

            contour = ax.contour(X1, X2, Y, levels=[gamma_j], colors=[colors_j], linewidths=3.0, linestyles=linestyles[0])
            handle_, _ = contour.legend_elements()
            handle_[0].set_linestyle(linestyles[0])
            labels.append("h_{}(x) = gamma_{}".format(j, j))
            handles.append(handle_[0])

            ax.contour(X1, X2, Y, levels=[lambda_j], colors=[colors_j], linewidths=3.0, linestyles=linestyles[1])
            handle_, _ = contour.legend_elements()
            handle_[0].set_linestyle(linestyles[1])
            labels.append("h_{}(x) = lambda_{}".format(j, j))
            handles.append(handle_[0])

        plt.xlabel('x_1')
        plt.ylabel('x_2')

        ax.legend(handles, labels)

        plt.title('State Trajectories')
        filename = 'jet_state_trajectories'
        if use_paper_results:
            filename += '_paper'
        plt.savefig('figures/{}.png'.format(filename))


if __name__ == "__main__":
    # read the configuration file
    with open("config.toml", "rb") as f:
        config = tomli.load(f)

    main(config)
