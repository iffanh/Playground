import numpy as np
import matplotlib.pyplot as plt

"""
The problem formulates a simple feedback MPC where we use the Riccati equation to 
obtain Kalman matrix for the control. 

x_{t+1} = A_t x_t + B_t u_t
u_t = -K_t x_t

K_t = R_t' B_t.T P_{t+1} (I + B_t R_t' B_t.T P_{t+1})' A_t
P_t = Q_t + A_t.T P_{t+1} (I + B_t R_t' B_t.T P_{t+1})' A_t
P_N = Q_N

In this particular instance we are only dealing with univariate problem

"""


def run_simple_feedback_mpc(N:int, a:float, b:float, q:float, r:float, x0:float, P_N:float):

    P_list = solve_riccati(N, P_N, a, b, q, r, [])
    K_list = calculate_gain(P_list, a, b, q, r)
    x_list, u_list = calculate_states(K_list, a, b, x0)


    states = {"P" : P_list, 
              "K" : K_list, 
              "x" : x_list, 
              "u" : u_list}
    plot_states(states)

    return


def plot_states(states:dict):

    fig, ax = plt.subplots(4, 1, figsize=(16,9))

    for i, key in enumerate(states.keys()):
        ax[i].plot(range(len(states[key])), states[key], marker='o', color='purple', label=key)
        ax[i].set_title(key)
        ax[i].legend()

    plt.show()

    return

def calculate_states(K_list:list, a:float, b:float, x0:float) -> tuple:

    x_list = []
    u_list = []

    x = x0
    x_list.append(x)
    for k in K_list:
        u = -k*x
        x = calculate_next_state(a, b, x, u)

        x_list.append(x)
        u_list.append(u)

    return x_list, u_list


def calculate_next_state(a:float, b:float, x_prev:float, u_curr:float):
    """
    Calculate the next state given the current state
    and the control
    """
    return a*x_prev + b*u_curr

def calculate_gain(P_list, a, b, q, r) -> list:

    K_list = []
    for p in P_list[1:]:
        K = b*p*a/(r*(1 + b*b*p/r))
        K_list.append(K)

    return K_list

def solve_riccati(ind, P, a, b, q, r, P_list) -> list:

    P_list.append(P)

    if ind == 0:
        return P_list[::-1]

    P = q + a*P*a/(1 + b*b*P/r)
    

    return solve_riccati(ind-1, P, a, b, q, r, P_list)


if __name__ == '__main__':

    # Define scalars
    a = 1.2
    b = 1
    N = 20
    q = 1 # State penalty coefficient
    r = 20 # Control penalty coefficient
    x0 = 1 # initial condition
    P_N = 1 # terminal condition

    run_simple_feedback_mpc(N, a, b, q, r, x0, P_N)