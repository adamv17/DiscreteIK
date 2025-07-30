import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import yaml
from scipy.optimize import minimize
import itertools

# =============================================================================
# Forward Kinematics & Core Math (Unchanged)
# =============================================================================
def fk(b_vals, robot_config, return_all_coords=False):
    """
    Computes the forward kinematics for the robot.
    """
    N = robot_config['joints_number']
    D = 2 * robot_config['ro']
    folded_len = robot_config['folded_cell_length']
    deployed_len = robot_config['deployed_cell_length']
    bend_angle = robot_config['bend_angle_const']
    bit_multiplier = D * np.sin(bend_angle)
    
    b = b_vals.reshape((2, N))
    # Clipping the argument of arcsin to the valid range [-1, 1]
    phi_arg = np.clip((b[0,:] - b[1,:]) * bit_multiplier / D, -1.0, 1.0)
    phi = np.arcsin(phi_arg)
    
    world_phi = np.cumsum(phi)
    b_avg = (b[0,:] + b[1,:]) / 2
    lengths = folded_len + (deployed_len - folded_len) * b_avg
    coords = np.zeros((N + 1, 2))
    unit_vectors = np.stack((np.sin(world_phi), np.cos(world_phi)), axis=1)
    segment_vectors = lengths[:, np.newaxis] * unit_vectors
    coords[1:, :] = np.cumsum(segment_vectors, axis=0)
    
    return coords if return_all_coords else coords[-1]

# =============================================================================
# Simplified Loss Calculation (No Analytical Gradient)
# =============================================================================

def _compute_fk_and_intermediates(b_vals, robot_config):
    """A helper to run FK and return values needed for the loss function."""
    N = robot_config['joints_number']
    D = 2 * robot_config['ro']
    folded_len = robot_config['folded_cell_length']
    deployed_len = robot_config['deployed_cell_length']
    bend_angle = robot_config['bend_angle_const']
    
    S_beta = np.sin(bend_angle)
    delta_L = deployed_len - folded_len

    b = b_vals.reshape((2, N))
    control_effort = b[0,:] - b[1,:]
    
    A = np.clip(S_beta * control_effort, -1.0, 1.0)
    phi = np.arcsin(A)

    world_phi = np.cumsum(phi)
    b_avg = (b[0,:] + b[1,:]) / 2
    lengths = folded_len + delta_L * b_avg
    coords = np.zeros((N + 1, 2))
    unit_vectors = np.stack((np.sin(world_phi), np.cos(world_phi)), axis=1)
    segment_vectors = lengths[:, np.newaxis] * unit_vectors
    coords[1:, :] = np.cumsum(segment_vectors, axis=0)

    return {
        "b": b, "control_effort": control_effort, "coords": coords
    }

def calculate_loss(b_vals, goal, robot_config, penalty_weights):
    """
    Computes the objective loss.
    This version includes penalties for distance, binariness, bend count, and smoothness.
    It does NOT compute a gradient.
    """
    N = robot_config['joints_number']
    
    epsilon = penalty_weights['epsilon']
    eta = penalty_weights['eta'] # Penalty for minimum bends
    kappa = penalty_weights['kappa'] # Penalty for smoothness (anti-zigzag)

    # Get the robot's state from the forward kinematics
    intermediates = _compute_fk_and_intermediates(b_vals, robot_config)
    x_N = intermediates['coords'][-1]
    control_effort = intermediates['control_effort']

    # 1. Distance to goal penalty
    dist_sq = np.sum(np.square(x_N - goal))
    
    # 2. Binary penalty (encourages values to be 0 or 1)
    reg_bin = -epsilon * np.sum(b_vals * (b_vals - 1.0))
    
    # 3. L1 penalty on control effort (encourages sparse bending)
    reg_num_bends = eta * np.sum(np.sqrt(control_effort**2 + 1e-9)) 
    
    # 4. Smoothness penalty (penalizes the second derivative of the control effort)
    reg_smoothness = 0.0
    if N > 2:
        # This penalizes sharp changes in bending between adjacent joints
        effort_double_prime = control_effort[2:] - 2 * control_effort[1:-1] + control_effort[:-2]
        reg_smoothness = kappa * np.sum(np.square(effort_double_prime))
    
    total_loss = dist_sq + reg_bin + reg_num_bends + reg_smoothness

    return total_loss

# =============================================================================
# Core Inverse Kinematics Logic (Hybrid SLSQP + Targeted Binary Search)
# =============================================================================
def is_configuration_valid(b_vals, obstacles_data, robot_config):
    """
    Checks if a given robot configuration collides with any obstacles.
    Returns True if valid (no collision), False otherwise.
    """
    if not obstacles_data:
        return True # No obstacles to collide with

    all_coords = fk(b_vals, robot_config, return_all_coords=True)
    
    for obstacle in obstacles_data:
        if obstacle['type'] == 'circle':
            center = np.array(obstacle['center'])
            radius = obstacle['radius']
            distances = np.linalg.norm(all_coords - center, axis=1) - radius - robot_config['ro']
            if np.min(distances) < 0:
                return False # Collision detected
        
        elif obstacle['type'] == 'rectangle':
            v1, v2 = obstacle['vertices']
            x_min, z_min = min(v1[0], v2[0]), min(v1[1], v2[1])
            x_max, z_max = max(v1[0], v2[0]), max(v1[1], v2[1])
            dx = np.maximum(x_min - all_coords[:, 0], all_coords[:, 0] - x_max)
            dz = np.maximum(z_min - all_coords[:, 1], all_coords[:, 1] - z_max)
            dist_outside = np.sqrt(np.maximum(dx, 0)**2 + np.maximum(dz, 0)**2)
            dist_inside = np.maximum(dx, dz)
            distances = np.where(dist_inside < 0, dist_inside, dist_outside) - robot_config['ro']
            if np.min(distances) < 0:
                return False # Collision detected
    
    return True # No collisions found

def discrete_inverse_kinematics(goal, obstacles_data, robot_config, ik_config):
    """
    Calculates optimal states using a hybrid SLSQP (exploration) and 
    a targeted binary search (refinement) strategy.
    The SLSQP optimizer relies on numerical gradient estimation.
    """
    N = robot_config['joints_number']
    max_iter = ik_config['max_iter']
    num_starts = ik_config['num_starts']
    num_critical_joints = ik_config.get('num_critical_joints', 5)

    # --- Define Simplified Penalty Weight Set ---
    # NOTE: You will need to add 'kappa_base' to your ik_config.yaml file
    penalty_weights = {
        'epsilon': (float(ik_config['epsilon_base']) / N),
        'eta': (float(ik_config['eta_base']) / N),
        'kappa': (float(ik_config['kappa_base']) / N)
    }

    def init_params():
        return np.clip(np.full(2 * N, 0.5) + (np.random.rand(2 * N) - 0.5) * 0.1, 0, 1)

    # --- Phase 1: Exploration with SciPy SLSQP ---
    print("--- Starting Phase 1: Exploration (SciPy SLSQP with Numerical Gradient) ---")
    best_exploratory_solution, best_exploratory_loss = None, np.inf
    
    loss_func = lambda b: calculate_loss(b, goal, robot_config, penalty_weights)

    bounds = [(0, 1) for _ in range(2 * N)]
    scipy_constraints = []
    for obstacle in obstacles_data:
        if obstacle['type'] == 'circle':
            center = np.array(obstacle['center'])
            radius = obstacle['radius']
            def circle_constraint_func(b_vals, c=center, r=radius):
                return np.min(np.linalg.norm(fk(b_vals, robot_config, True) - c, axis=1) - r - robot_config['ro'])
            scipy_constraints.append({'type': 'ineq', 'fun': circle_constraint_func})
        
        elif obstacle['type'] == 'rectangle':
            v1, v2 = obstacle['vertices']
            x_min, z_min = min(v1[0], v2[0]), min(v1[1], v2[1])
            x_max, z_max = max(v1[0], v2[0]), max(v1[1], v2[1])
            def rect_constraint_func(b_vals, r_xmin=x_min, r_xmax=x_max, r_zmin=z_min, r_zmax=z_max):
                all_coords = fk(b_vals, robot_config, return_all_coords=True)
                dx = np.maximum(r_xmin - all_coords[:, 0], all_coords[:, 0] - r_xmax)
                dz = np.maximum(r_zmin - all_coords[:, 1], all_coords[:, 1] - r_zmax)
                dist_outside = np.sqrt(np.maximum(dx, 0)**2 + np.maximum(dz, 0)**2)
                dist_inside = np.maximum(dx, dz)
                return np.min(np.where(dist_inside < 0, dist_inside, dist_outside) - robot_config['ro'])
            scipy_constraints.append({'type': 'ineq', 'fun': rect_constraint_func})

    optimizer_options = {'maxiter': max_iter, 'ftol': 1e-6, 'disp': False}

    for i in range(num_starts):
        print(f"\n--- Exploration Run {i+1}/{num_starts} ---")
        initial_guess = init_params()
        
        current_result = minimize(loss_func, initial_guess, method='SLSQP', 
                                  options=optimizer_options, bounds=bounds, constraints=scipy_constraints)

        if current_result.success:
            current_loss = current_result.fun
            print(f"Run {i+1} successful. Loss: {current_loss:.4f}")
            if current_loss < best_exploratory_loss:
                best_exploratory_loss = current_loss
                best_exploratory_solution = current_result.x
                print(f"*** New best exploratory solution found (Loss: {best_exploratory_loss:.4f})! ***")
                
                if best_exploratory_loss < 1.0:
                    print("\nExcellent solution found! Proceeding directly to refinement.")
                    break
        else:
            print(f"Run {i+1} did not converge. Message: {current_result.message}")

    print(f"\n--- Phase 1 Complete. Best Loss Found: {best_exploratory_loss:.4f} ---")

    if best_exploratory_solution is None:
        print("Exploration failed to find any valid solution.")
        return None, None, None

    plot_results(best_exploratory_solution, np.round(best_exploratory_solution), goal, obstacles_data, robot_config, 
                 title="After Exploration Phase (SLSQP)")

    # --- Phase 2: Targeted Binary Search (with Collision Checking) ---
    print(f"\n--- Starting Phase 2: Targeted Binary Search with Collision Checking ---")
    
    non_binary_scores = np.abs(best_exploratory_solution - 0.5)
    critical_indices = np.argsort(non_binary_scores)[:num_critical_joints]
    print(f"Identified {len(critical_indices)} critical actuators to search.")

    base_solution = np.round(best_exploratory_solution)
    
    best_binary_solution = None
    best_binary_dist = np.inf

    # Check if the initial rounded solution is valid
    if is_configuration_valid(base_solution, obstacles_data, robot_config):
        best_binary_solution = base_solution.copy()
        best_binary_dist = np.linalg.norm(fk(best_binary_solution, robot_config) - goal)
        print(f"Initial rounded solution is valid. Distance: {best_binary_dist:.4f}")
    else:
        print("Initial rounded solution is invalid (collides with obstacle).")

    num_combinations = 2**len(critical_indices)
    print(f"Testing {num_combinations} binary combinations...")
    valid_solutions_found = 0

    for i, combination in enumerate(itertools.product([0, 1], repeat=len(critical_indices))):
        if i % 100 == 0:
            print(f"\rTesting combination {i+1}/{num_combinations}", end="")
        
        test_solution = base_solution.copy()
        test_solution[critical_indices] = combination
        
        # Crucial Step: Check if this binary configuration is valid
        if is_configuration_valid(test_solution, obstacles_data, robot_config):
            dist = np.linalg.norm(fk(test_solution, robot_config) - goal)
            
            if dist < best_binary_dist:
                valid_solutions_found += 1
                best_binary_dist = dist
                best_binary_solution = test_solution.copy()
            
    print(f"\n--- Phase 2 Complete. Found {valid_solutions_found} valid combinations. ---")
    
    if best_binary_solution is None:
        print("\nWARNING: Binary search could not find any valid, collision-free solution.")
        print("Returning the continuous solution from Phase 1. The binary result will be invalid.")
        # Fallback to a potentially invalid solution for visualization
        best_binary_solution = np.round(best_exploratory_solution)

    print(f"Best valid binary distance: {best_binary_dist:.4f}")

    res_continuous = best_exploratory_solution
    res_bits = best_binary_solution
    
    # --- Final Processing ---
    b_reshaped = res_bits.reshape((2, N))
    action_states = np.zeros(N, dtype=int)
    for idx in range(N):
        b1, b2 = int(b_reshaped[0, idx]), int(b_reshaped[1, idx])
        if b1 == 0 and b2 == 0: action_states[idx] = 0
        elif b1 == 1 and b2 == 1: action_states[idx] = 1
        elif b1 == 0 and b2 == 1: action_states[idx] = 2
        elif b1 == 1 and b2 == 0: action_states[idx] = 3

    print("\nHybrid optimization complete.")
    return action_states, res_bits, res_continuous

# =============================================================================
# Visualization (Unchanged)
# =============================================================================
def plot_results(b_continuous, b_binary, goal, obstacles_data, config, title="Soft Robot Configuration Analysis"):
    """Plots the robot configuration with a customizable title."""
    if b_binary is None or b_continuous is None: 
        print("Cannot plot results, a solution was not found.")
        return
    coords_continuous = fk(b_continuous, config, return_all_coords=True)
    coords_binary = fk(b_binary, config, return_all_coords=True)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.plot(coords_continuous[:, 0], coords_continuous[:, 1], 'b-o', markersize=3, linewidth=2, label='Optimal (Continuous)')
    ax.plot(coords_binary[:, 0], coords_binary[:, 1], 'g--s', markersize=3, linewidth=2, label='Optimal (Binary)')
    ax.plot(coords_continuous[-1, 0], coords_continuous[-1, 1], 'b*', markersize=15, label=f'Endpoint (Continuous): ({coords_continuous[-1][0]:.2f}, {coords_continuous[-1][1]:.2f})')
    ax.plot(coords_binary[-1, 0], coords_binary[-1, 1], 'gP', markersize=12, label=f'Endpoint (Binary): ({coords_binary[-1][0]:.2f}, {coords_binary[-1][1]:.2f})')
    ax.plot(goal[0], goal[1], 'rX', markersize=15, label=f'Goal: ({goal[0]}, {goal[1]})')
    goal_radius = plt.Circle(goal, config['end_effector_detection_radius'], color='red', fill=False, linestyle='--', alpha=0.5, label='Goal Radius')
    ax.add_artist(goal_radius)
    for i, obs in enumerate(obstacles_data):
        label = 'Obstacle' if i == 0 else None
        if obs['type'] == 'circle':
            circle = patches.Circle(obs['center'], obs['radius'], color='r', fill=True, alpha=0.3, label=label)
            ax.add_patch(circle)
        elif obs['type'] == 'rectangle':
            v1, v2 = obs['vertices']
            x_min, z_min = min(v1[0], v2[0]), min(v1[1], v2[1])
            width, height = abs(v1[0] - v2[0]), abs(v1[1] - v2[1])
            rect = patches.Rectangle((x_min, z_min), width, height, color='r', fill=True, alpha=0.3, label=label)
            ax.add_patch(rect)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('X Coordinate (mm)', fontsize=12)
    ax.set_ylabel('Z Coordinate (mm)', fontsize=12)
    ax.legend(fontsize=10)
    ax.axis('equal')
    plt.show()

# =============================================================================
# Main Execution Block (Unchanged)
# =============================================================================
if __name__ == "__main__":
    try:
        with open('robot_config.yaml', 'r') as file:
            robot_config = yaml.safe_load(file)
        with open('env.yaml', 'r') as file:
            env_config = yaml.safe_load(file)
        with open('ik_config.yaml', 'r') as file:
            ik_config = yaml.safe_load(file)
    except FileNotFoundError as e:
        print(f"Error: Configuration file not found. {e}")
        exit()
    
    GOAL = np.array(env_config.get('goal', [400, 600]))
    OBSTACLES = env_config.get('obstacles', [])

    action_sequence, res_bits, res_continuous = discrete_inverse_kinematics(GOAL, OBSTACLES, robot_config, ik_config)

    if action_sequence is not None:
        print("\n" + "="*30)
        print("           Final Results")
        print("="*30)
        final_pos = fk(res_bits, robot_config)
        print(f"Goal Position: {GOAL}")
        print(f"Final Position (Binary): [{final_pos[0]:.4f}, {final_pos[1]:.4f}]")
        print(f"Final Distance to Goal: {np.linalg.norm(final_pos - GOAL):.4f}")
        print("\n--- Action Sequence ---")
        print("0:closed, 1:open, 2:left, 3:right")
        print(action_sequence)
        print("="*30)
        plot_results(res_continuous, res_bits, GOAL, OBSTACLES, robot_config, 
                     title="After Final Refinement")
