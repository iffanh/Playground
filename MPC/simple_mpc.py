import numpy as np
import matplotlib.pyplot as plt

"""
The problem is formulated as follows: given the scalar system

x_{t+1} = ax_t + bu_t

where a and b are scalars. The objective function is defined as

f(z) = sum_{t=0}^{N-1} 1/2 q (x_{t+1})^2 + 1/2 r (u_{t})^2

where q and r are scalars.

"""

def run_simple_mpc(N, a, b, q, r, x0, maxIter = 10):
    """
    Main function
    """

    # Build the matrix
    A = build_matrix(N, a, b)
    A_pinv = np.linalg.pinv(A)

    x_all_sol = [] # All the solution of the optimization problem
    u_all_sol = []
    x_sol = [] # Only for the first solution of each timestep
    u_sol = []
    for i in range(maxIter):
        B = build_vector(N, a, x0)
        z = np.dot(A_pinv, B)
        x_all_sol.append(z[:N+1, 0])
        u_all_sol.append(z[N+1:, 0])
        x_sol.append(z[0, 0])
        u_sol.append(z[N+1, 0])
        x0 = calculate_next_state(a, b, x0, z[N+1, 0])
        x0 = z[0, 0]

    plot_solution(x_all_sol, u_all_sol, x_sol, u_sol)

def plot_solution(x_all_sol, u_all_sol, x_sol, u_sol):
    """
    Function only to plot the solution
    """

    fig, ax = plt.subplots(2, 1, figsize=(16,9))

    for i in range(len(x_sol)):
        ax[0].plot(range(i, N+1+i), x_all_sol[i], color='blue')
        ax[1].plot(range(i, N+1+i), u_all_sol[i], color='red')

    ax[0].plot(range(len(x_sol)), x_sol, marker='o', color='purple', label="Solution")
    ax[1].plot(range(len(u_sol)), u_sol, marker='o', color='purple', label="Solution")

    ax[0].set_title("State")
    ax[1].set_title("Control")

    ax[0].legend()
    ax[1].legend()

    plt.show(fig)
    return

def calculate_next_state(a:float, b:float, x_prev:float, u_curr:float):
    """
    Calculate the next state given the current state
    and the control
    """
    return a*x_prev + b*u_curr

def build_vector(N:int, a:float, x0:float):
    """
    Function responsible for constructing the b vector
    in Az = B
    """
    vec = np.zeros((N+1, 1))
    vec[0,0] = a*x0
    return vec

def build_matrix(N:int, a:float, b:float):
    """
    Function responsible for constructing the A matrix
    in Az = B
    """

    # initialize matrix
    mat = np.zeros((N+1, 2*(N+1)))
    tmp = np.diag([-b]*(N+1))

    tmp1 = np.diag([1]*(N+1))
    tmp2 = np.diag([-a]*N)

    mat[:N+1, :N+1] = tmp1
    mat[1:N+1, 0:N] += tmp2
    mat[:, N+1:] = tmp

    return mat

if __name__ == '__main__':

    # Define scalars
    a = 1.2
    b = 0.4
    N = 4
    q = 1
    r = 4
    x0 = 10 # initial condition

    run_simple_mpc(N, a, b, q, r, x0)