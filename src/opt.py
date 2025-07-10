import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os
from datetime import datetime

# ----------------------------
#           Parameters
# ----------------------------

N = 78              # Number of frusta
D = 4               # Frusta Diameter
closed_len = 1/3.   # Length of closed frusta (l0)

epsilon = (4e1)/N   # Binary Regularization Term (greater than 0)
zeta = (1e-1)/N     # Penalize opening a single frustum instead of in groups
xi = (5e-1)/N       # Penalize Bending

max_iter = 100      # Max optimizer iterations
num_starts = 10     # Number of optimization runs (multi-start)

# ----------------------------
#           Goal
# ----------------------------

GOAL = np.array([N-10, 10])

# ----------------------------
#       Forward Kinematics
# ----------------------------

def calc_x(index, origin, b_avg, theta):
    """Recursive function to calculate the position of the end-effector."""
    if index == N:
        return origin
    unit = np.array([np.cos(theta[index]), np.sin(theta[index])])
    length = (1 - closed_len) * b_avg[index] + closed_len
    x = origin + length * unit
    return calc_x(index + 1, x, b_avg, theta)

def world_coord(theta):
    """Calculates the cumulative sum of angles to get world coordinates."""
    # This function modifies theta in-place by creating cumulative sums
    return np.cumsum(theta)

def fk(b_vals):
    """Computes the forward kinematics to find the end-effector position."""
    b = b_vals.reshape((2, N))
    # Using a copy prevents modification of the original array by world_coord
    theta = np.arcsin(np.clip((b[0,:] - b[1,:]) / D, -1.0, 1.0))
    world_theta = world_coord(theta)
    b_avg = (b[0,:] + b[1,:]) / 2
    x_N = calc_x(0, np.array([0,0]), b_avg, world_theta)
    return x_N


# ----------------------------
#         Loss Function
# ----------------------------

def loss(b_vals):
    """Calculates the total loss for the optimization problem."""
    x_N = fk(b_vals)
    # Use squared norm as per the objective function
    dist_sq = np.sum(np.square(x_N - GOAL))
    
    # Binary regularization
    reg_bin = -epsilon * np.sum(b_vals * (b_vals - 1.0))

    b = b_vals.reshape((2, N))
    
    # Grouping penalty for 'open' state
    openness = np.prod(b, axis=0) # Calculates b1,i * b2,i for each frustum i
    reg_group = zeta * np.sum(np.square(openness[:-1] - openness[1:]))

    # Bending penalty
    theta = np.arcsin(np.clip((b[0,:] - b[1,:]) / D, -1.0, 1.0))
    reg_bend = xi * np.sum(np.square(theta))

    return dist_sq + reg_bin + reg_group + reg_bend

# ----------------------------
#         Constraints
# ----------------------------

# Bounds are more efficiently handled by the optimizer's `bounds` argument
bounds = [(0, 1) for _ in range(2 * N)]

# ----------------------------
#     Initialize Opt. Params.
# ----------------------------

def init_params():
    """Generates a random initial guess for the actuator coefficients."""
    return np.random.rand(2 * N)

# ----------------------------
#         Optimization
# ----------------------------

optimizer_options = {'maxiter': max_iter, 'ftol': 1e-8, 'disp': False}

best_result = None
best_rounded_loss = np.inf

print(f"Starting multi-start optimization with {num_starts} runs...")
print("NOTE: Selecting best solution based on the loss from ROUNDED coefficients.")

for i in range(num_starts):
    print(f"\n--- Optimization Run {i+1}/{num_starts} ---")
    initial_guess = init_params()
    
    # Using 'L-BFGS-B' or 'TNC' can be more efficient for simple box bounds
    current_result = minimize(loss, x0=initial_guess, method='SLSQP', options=optimizer_options, bounds=bounds)

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
        print(f"Run {i+1} did not converge. Message: {current_result.message}")

# Check if any successful optimization was found
if best_result is None:
    raise RuntimeError("Optimization failed to find a solution across all starts. Try increasing max_iter or num_starts.")

print("\nMulti-start optimization complete. Using best result for analysis.")

# ----------------------------
#          Analysis
# ----------------------------

print("\n" + "="*30)
print("      Optimization Results")
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
    theta = np.arcsin(np.clip((b[0, :] - b[1, :]) / D, -1.0, 1.0))
    world_theta = world_coord(theta) # Use a copy to be safe
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
    f.write("      Optimization Log\n")
    f.write("="*30 + "\n")
    f.write(f"Run Timestamp: {timestamp}\n\n")
    
    f.write("--- Parameters ---\n")
    f.write(f"Number of frusta (N): {N}\n")
    f.write(f"Binary Regularization (epsilon): {epsilon:.6f}\n")
    f.write(f"Grouping Penalty (zeta):       {zeta:.6f}\n")
    f.write(f"Bending Penalty (xi):          {xi:.6f}\n")
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
