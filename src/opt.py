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
# --- Penalty Tuning ---
zeta = (1e-1)/N     # Penalize opening a single frustum instead of in groups
# Increased bending penalty to discourage any bending more strongly.
xi = (5e0)/N        # Penalize Bending
# New penalty to enforce smooth curves and directly penalize zig-zagging.
kappa = (1e0)/N # Penalize changes in curvature between frustums. Set to 0 to disable.


max_iter = 100      # Max optimizer iterations
num_starts = 10     # Number of optimization runs (multi-start)

# ----------------------------
#           Goal
# ----------------------------

GOAL = np.array([N-10, 10])

# ----------------------------
#       Forward Kinematics
# ----------------------------

def calc_x_iterative(b_avg, world_theta):
    """Iterative function to calculate the position of all joints."""
    coords = np.zeros((N + 1, 2))
    unit_vectors = np.stack((np.cos(world_theta), np.sin(world_theta)), axis=1)
    lengths = (1 - closed_len) * b_avg + closed_len
    segment_vectors = lengths[:, np.newaxis] * unit_vectors
    coords[1:, :] = np.cumsum(segment_vectors, axis=0)
    return coords

def world_coord(theta):
    """Calculates the cumulative sum of angles to get world coordinates."""
    return np.cumsum(theta)

def fk(b_vals):
    """Computes the forward kinematics to find the end-effector position."""
    b = b_vals.reshape((2, N))
    theta = np.arcsin(np.clip((b[0,:] - b[1,:]) / D, -1.0, 1.0))
    world_theta = world_coord(theta)
    b_avg = (b[0,:] + b[1,:]) / 2
    
    # Using the more efficient iterative calculation
    all_coords = calc_x_iterative(b_avg, world_theta)
    return all_coords[-1]


# ----------------------------
#         Loss Function
# ----------------------------

def loss(b_vals):
    """Calculates the total loss for the optimization problem."""
    x_N = fk(b_vals)
    dist_sq = np.sum(np.square(x_N - GOAL))
    
    reg_bin = -epsilon * np.sum(b_vals * (b_vals - 1.0))

    b = b_vals.reshape((2, N))
    
    openness = np.prod(b, axis=0)
    reg_group = zeta * np.sum(np.square(openness[:-1] - openness[1:]))

    theta = np.arcsin(np.clip((b[0,:] - b[1,:]) / D, -1.0, 1.0))
    reg_bend = xi * np.sum(np.square(theta))

    reg_smooth = kappa * np.sum(np.square(theta[:-1] - theta[1:]))

    return dist_sq + reg_bin + reg_group + reg_bend + reg_smooth

# ----------------------------
#         Constraints
# ----------------------------

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
    
    current_result = minimize(loss, x0=initial_guess, method='SLSQP', options=optimizer_options, bounds=bounds)

    if current_result.success:
        current_res_bits = np.round(current_result.x)
        current_rounded_loss = loss(current_res_bits)
        
        print(f"Run {i+1} successful. Continuous Loss: {current_result.fun:.4f}, Rounded Loss: {current_rounded_loss:.4f}")

        if current_rounded_loss < best_rounded_loss:
            best_rounded_loss = current_rounded_loss
            best_result = current_result
            print(f"*** New best solution found (Rounded Loss: {best_rounded_loss:.4f})! ***")
    else:
        print(f"Run {i+1} did not converge. Message: {current_result.message}")

if best_result is None:
    raise RuntimeError("Optimization failed to find a solution across all starts. Try increasing max_iter or num_starts.")

print("\nMulti-start optimization complete. Using best result for analysis.")

# ----------------------------
#          Analysis
# ----------------------------

print("\n" + "="*30)
print("      Optimization Results")
print("="*30)

b_star = best_result.x
x_N_star = fk(b_star)
res_bits = np.round(b_star)
x_N_bits = fk(res_bits)

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

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
base_filename = f"run_{timestamp}"
log_filepath = os.path.join(log_dir, f"{base_filename}.txt")
img_filepath = os.path.join(log_dir, f"{base_filename}.png")

# --- Graphical Analysis ---

def get_all_coords(b_vals):
    """Calculates the coordinates of each joint in the frusta chain."""
    b = b_vals.reshape((2, N))
    theta = np.arcsin(np.clip((b[0, :] - b[1, :]) / D, -1.0, 1.0))
    world_theta = world_coord(theta)
    b_avg = (b[0, :] + b[1, :]) / 2
    
    # Use the efficient iterative FK calculation
    return calc_x_iterative(b_avg, world_theta)

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
    f.write(f"Smoothness Penalty (lambda):   {kappa:.6f}\n") 
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

plot_and_save_results(b_star, res_bits, img_filepath)
