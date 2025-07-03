import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os
from datetime import datetime

# ----------------------------
#           Parameters
# ----------------------------

N = 150             # Number of frusta
epsilon = (2e1)/N   # Regularization term (greater than 0)
max_iter = 100      # Max optimizer iterations
D = 4               # Frusta Diameter
closed_len = 1/3.   # Length of closed frusta (l0)
num_starts = 10     # Number of optimization runs (multi-start)

# ----------------------------
#           Goal
# ----------------------------

GOAL = np.array([N-30, 30])

# ----------------------------
#       Forward Kinematics
# ----------------------------

def calc_x(index, origin, b_avg, theta):
    if index == N:
        return origin
    unit = np.array([np.cos(theta[index]), np.sin(theta[index])])
    x = origin + (1-closed_len) * b_avg[index] * unit + closed_len * unit
    return calc_x(index+1, x, b_avg, theta)

def world_coord(theta):
    # This function modifies theta in-place by creating cumulative sums
    for i in range(1,N):
        theta[i] = theta[i] + theta[i-1]
    return theta

def fk(b_vals):
    b = b_vals.reshape((2, N))
    # Using a copy prevents modification of the original array by world_coord
    theta = np.arcsin((b[0,:] - b[1,:]) / D)
    world_theta = world_coord(theta.copy())
    x_N = calc_x(0, np.array([0,0]), (b[0,:] + b[1,:]) / 2, world_theta)
    return x_N


# ----------------------------
#         Loss Function
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
#     Initialize Opt. Params.
# ----------------------------

def init_params():
    return np.random.rand(2 * N)

# ----------------------------
#         Optimization
# ----------------------------

optimizer_options = {'maxiter': max_iter, 'ftol': 1e-8, 'disp': False} # disp is False to avoid clutter

best_result = None
best_rounded_loss = np.inf

print(f"Starting multi-start optimization with {num_starts} runs...")
print("NOTE: Selecting best solution based on the loss from ROUNDED coefficients.")

for i in range(num_starts):
    print(f"\n--- Optimization Run {i+1}/{num_starts} ---")
    initial_guess = init_params()
    
    current_result = minimize(loss, x0=initial_guess, method='SLSQP', options=optimizer_options, constraints=constraints)

    if current_result.success:
        # Calculate the loss for the rounded result of the current run
        current_res_bits = np.round(current_result.x)
        current_rounded_loss = loss(current_res_bits)
        
        print(f"Run {i+1} successful. Continuous Loss: {current_result.fun:.4f}, Rounded Loss: {current_rounded_loss:.4f}")

        # The core change: compare based on the rounded loss
        if current_rounded_loss < best_rounded_loss:
            best_rounded_loss = current_rounded_loss
            best_result = current_result
            print(f"*** New best solution found (Rounded Loss: {best_rounded_loss:.4f})! ***")
    else:
        print(f"Run {i+1} did not converge.")

# Check if any successful optimization was found
if best_result is None:
    raise RuntimeError("Optimization failed to find a solution across all starts. Try increasing max_iter or num_starts.")

print("\nMulti-start optimization complete. Using best result for analysis.")

# ----------------------------
#           Analysis
# ----------------------------

print("\n" + "="*30)
print("     Optimization Results")
print("="*30)

# --- Calculations ---
# Use the best result found from all the runs
b_star = best_result.x
x_N_star = fk(b_star)
res_bits = np.round(b_star)
x_N_bits = fk(res_bits)

# --- Print Results ---
print(f"Goal: {GOAL}")
print(f"\nFinal Loss (Unrounded): {loss(b_star):.4f}")
print(f"Endpoint (Unrounded): {x_N_star}")
print(f"\nFinal Loss (Rounded):   {loss(res_bits):.4f}")
print(f"Endpoint (Rounded):   {x_N_bits}")
print("="*30)


# --- File Logging Setup ---
log_dir = "log"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    print(f"Created directory: {log_dir}")

# Generate a unique filename based on the current time
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
base_filename = f"run_{timestamp}"
log_filepath = os.path.join(log_dir, f"{base_filename}.txt")
img_filepath = os.path.join(log_dir, f"{base_filename}.png")

# --- Graphical Analysis ---

def get_all_coords(b_vals):
    """Calculates the coordinates of each joint in the frusta chain."""
    b = b_vals.reshape((2, N))
    theta = np.arcsin((b[0, :] - b[1, :]) / D)
    world_theta = world_coord(theta.copy()) # Use a copy to be safe
    b_avg = (b[0, :] + b[1, :]) / 2

    coords = np.zeros((N + 1, 2))
    current_pos = np.array([0., 0.])
    coords[0, :] = current_pos

    for i in range(N):
        unit = np.array([np.cos(world_theta[i]), np.sin(world_theta[i])])
        length = (1 - closed_len) * b_avg[i] + closed_len
        current_pos = current_pos + length * unit
        coords[i+1, :] = current_pos
    return coords

def plot_and_save_results(b_continuous, b_binary, filename):
    """Plots the robot configuration and saves it to a file."""
    coords_continuous = get_all_coords(b_continuous)
    coords_binary = get_all_coords(b_binary)

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 10))

    plt.plot(coords_continuous[:, 0], coords_continuous[:, 1],
             'b-o', markersize=3, linewidth=2, label='Optimal (Continuous)')
    plt.plot(coords_binary[:, 0], coords_binary[:, 1],
             'g--s', markersize=3, linewidth=2, label='Optimal (Binary)')
    
    plt.plot(coords_continuous[-1, 0], coords_continuous[-1, 1],
             'b*', markersize=15, label=f'Endpoint (Continuous): ({coords_continuous[-1][0]:.2f}, {coords_continuous[-1][1]:.2f})')
    plt.plot(coords_binary[-1, 0], coords_binary[-1, 1],
             'gP', markersize=12, label=f'Endpoint (Binary): ({coords_binary[-1][0]:.2f}, {coords_binary[-1][1]:.2f})')
    plt.plot(GOAL[0], GOAL[1],
             'rX', markersize=15, label=f'Goal: ({GOAL[0]}, {GOAL[1]})')

    plt.title('Soft Robot Configuration Analysis', fontsize=16)
    plt.xlabel('X Coordinate', fontsize=12)
    plt.ylabel('Y Coordinate', fontsize=12)
    plt.legend(fontsize=10)
    plt.axis('equal')
    
    # Save the figure to the specified path
    plt.savefig(filename)
    print(f"Figure saved to: {filename}")
    plt.show()


# --- Save Text Log File ---
with open(log_filepath, 'w') as f:
    f.write("="*30 + "\n")
    f.write("   Optimization Log\n")
    f.write("="*30 + "\n")
    f.write(f"Run Timestamp: {timestamp}\n\n")
    
    f.write("--- Parameters ---\n")
    f.write(f"Number of frusta (N): {N}\n")
    f.write(f"Regularization (epsilon): {epsilon}\n")
    f.write(f"Max Iterations: {max_iter}\n")
    f.write(f"Number of Starts: {num_starts}\n")
    f.write(f"Goal: {GOAL}\n\n")

    f.write("--- Results ---\n")
    f.write(f"Final Loss (Unrounded): {loss(b_star):.6f}\n")
    f.write(f"Endpoint (Unrounded): {x_N_star}\n\n")
    f.write(f"Final Loss (Rounded):   {loss(res_bits):.6f}\n")
    f.write(f"Endpoint (Rounded):   {x_N_bits}\n\n")
    
    f.write("--- Optimal Coefficients (Continuous) ---\n")
    np.savetxt(f, b_star, fmt='%.8f')
    f.write("\n")

    f.write("--- Optimal Coefficients (Rounded) ---\n")
    np.savetxt(f, res_bits, fmt='%d')

print(f"Log file saved to: {log_filepath}")

# Generate and save the plot
plot_and_save_results(b_star, res_bits, img_filepath)