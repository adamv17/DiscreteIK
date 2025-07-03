import numpy as np
from scipy.optimize import minimize

# ---------------------------- 
#          Parameters          
# ---------------------------- 

N = 150             # Number of frusta 
epsilon = (1e-3)*N  # Regularization term (greater than 0)
max_iter = 100      # Max optimizer iterations 
D = 4               # Frusta Diameter
closed_len = 1/3.   # Length of closed frusta (l0)  

# ---------------------------- 
#            Goal          
# ---------------------------- 

GOAL = np.array([N, 0])

# ---------------------------- 
#      Forward Kinematics          
# ---------------------------- 

def calc_x(index, origin, b_avg, theta):
    if index == N:
        return origin
    unit = np.array([np.cos(theta[index]), np.sin(theta[index])])
    x = origin + (1-closed_len) * b_avg[index] * unit + closed_len * unit
    return calc_x(index+1, x, b_avg, theta)

def world_coord(theta):
    for i in range(1,N):
        theta[i] = theta[i] + theta[i-1]
    return theta

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
    dist = np.linalg.norm(x_N - GOAL, ord=2)
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

optimizer_options = {'maxiter': max_iter, 'ftol': 1e-8, 'disp': True}
initial_guess = init_params()

result = minimize(loss, x0=initial_guess, method='SLSQP', options=optimizer_options, constraints=constraints)

# ---------------------------- 
#          Analysis          
# ---------------------------- 

b_star = result.x.reshape((2, N))
x_N_star = fk(b_star)

print("Goal:", GOAL)
print("Optimization Results")
print("Optimal Coefficients:", b_star)
print("Endpoint:", x_N_star)
print("Final Loss:", loss(result.x))
print(fk(np.ones(2*N)))  # For testing with all ones