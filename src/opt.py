import numpy as np
from scipy.optimize import minimize

# ---------------------------- 
#          Parameters          
# ---------------------------- 

N = 5               # Number of frusta 
epsilon = 10        # Regularization term (greater than 0)
max_iter = 100      # Max optimizer iterations 
D = 4               # Frusta Diameter
closed_len = 1/3.   # Length of closed frusta (l0)  

# ---------------------------- 
#            Goal          
# ---------------------------- 

GOAL = np.array([100, 100])

# ---------------------------- 
#      Forward Kinematics          
# ---------------------------- 

def calc_x(index, origin, b_avg, theta):
    if index == N:
        return origin
    unit = np.array([np.cos(theta[index]), np.sin(theta[index])])
    x = origin + b_avg * unit + closed_len * unit
    return calc_x(index+1, x, b_avg, theta)

def world_coord(theta):
    for i,t in enumerate(theta, start=1):
        theta[i] = t + theta[i-1]

def fk(b_vals):
    b = b_vals.reshape((2, N))
    theta = np.arcsin((b[0,:] - b[1,:]) / D)
    world_theta = world_coord(theta)
    x_N = calc_x(0, np.array([0,0]), (b[0,:] + b[1,:]) / 2, world_theta)
    return x_N


# ---------------------------- 
#        Loss Function          
# ---------------------------- 

def loss(b_vals):
    x_N = fk(b_vals)
    dist = np.linalg.norm(x_N - GOAL, order=2)
    reg = -epsilon * np.sum(b_vals * (b_vals - np.ones(2*N)))
    return dist + reg

# ---------------------------- 
#         Constraints           
# ---------------------------- 

def ineq_lower(b_vals):
    return b_vals

def ineq_upper(b_vals):
    return np.ones(2*N) - b_vals 

# 0 <= b <= 1
constraints = [{'type': 'ineq', 'fun': ineq_lower}, {'type': 'ineq', 'fun': ineq_upper}]

# ---------------------------- 
#   Initialize Opt. Params.          
# ---------------------------- 

def init_params():
    return np.random.rand(2 * N)

# ---------------------------- 
#        Optimization          
# ---------------------------- 

optimizer_options = {'maxiter': max_iter, 'ftol': 1e-6, 'gtol': 1e-8, 'disp': False}
initial_guess = init_params()

result = minimize(loss, x0=initial_guess, method='L-BFGS-B', options=optimizer_options, constraints=constraints)

# ---------------------------- 
#          Analysis          
# ---------------------------- 

print(result.x)