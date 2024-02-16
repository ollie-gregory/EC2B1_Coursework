# Define regression function
import numpy as np

def regression_coefs(Y, *args):
    T = len(Y)

    X = np.concatenate([arg[:, None] for arg in args], axis=1)

    XX = X.T @ X
    XY = X.T @ Y

    coefs = np.linalg.inv(XX) @ XY
    return coefs

def calc_add_lin(var):
    Y = var
    T = len(Y)

    # Additive linear Model: y = a + bt + u
    x1, x2 = [np.empty(T) for j in range(2)]

    for t in range(T):
        x1[t] = 1.
        x2[t] = t + 1

    a_add_lin, b_add_lin = regression_coefs(Y, x1, x2)

    add_lin_reg = np.empty(T)

    for t in range(T):
        add_lin_reg[t] = a_add_lin + b_add_lin * (t + 1)
    
    return add_lin_reg

def calc_add_quad(var):
    Y = var
    T = len(Y)

    # Additive quadratic Model: y = a + bt + ct^2 + u
    x1, x2, x3 = [np.empty(T) for j in range(3)]

    for t in range(T):
        x1[t] = 1.
        x2[t] = t + 1
        x3[t] = (t + 1) ** 2

    a_add_quad, b1_add_quad, b2_add_quad = regression_coefs(Y, x1, x2, x3)

    add_quad_reg = np.empty(T)

    for t in range(T):
        add_quad_reg[t] = a_add_quad + b1_add_quad * (t + 1) + b2_add_quad * ((t + 1) ** 2)
    
    return add_quad_reg

def calc_exp_lin(var):
    Y = var
    T = len(Y)

    # Exponential linear Model: y = a + b * exp(c * t) + u
    x1, x2 = [np.empty(T) for j in range(2)]

    for t in range(T):
        x1[t] = 1.
        x2[t] = t + 1

    a_exp_lin, b_exp_lin = regression_coefs(np.log(Y), x1, x2)

    exp_lin_reg = np.empty(T)

    for t in range(T):
        exp_lin_reg[t] = a_exp_lin + b_exp_lin * (t-1)
    
    return exp_lin_reg

def calc_exp_quad(var):
    Y = var
    T = len(Y)

    # Exponential quadratic Model: y = a + b * exp(c * t) + u
    x1, x2, x3 = [np.empty(T) for j in range(3)]

    for t in range(T):
        x1[t] = 1.
        x2[t] = t + 1
        x3[t] = (t + 1) ** 2

    a_exp_quad, b1_exp_quad, b2_exp_quad = regression_coefs(np.log(Y), x1, x2, x3)

    exp_quad_reg = np.empty(T)

    for t in range(T):
        exp_quad_reg[t] = a_exp_quad + b1_exp_quad * (t-1) + b2_exp_quad * ((t-1) ** 2)
    
    return exp_quad_reg