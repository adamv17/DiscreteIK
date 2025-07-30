import numpy as np
import nlopt
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
    phi = np.arcsin(np.clip((b[0,:] - b[1,:]) * bit_multiplier / D, -1.0, 1.0))
    world_phi = np.cumsum(phi)
    b_avg = (b[0,:] + b[1,:]) / 2
    lengths = folded_len + (deployed_len - folded_len) * b_avg
    coords = np.zeros((N + 1, 2))
    unit_vectors = np.stack((np.sin(world_phi), np.cos(world_phi)), axis=1)
    segment_vectors = lengths[:, np.newaxis] * unit_vectors
    coords[1:, :] = np.cumsum(segment_vectors, axis=0)
    
    return coords if return_all_coords else coords[-1]

# =============================================================================
# Analytical Gradient Calculation Functions
# =============================================================================

def calculate_loss_and_analytical_gradient(b_vals, goal, robot_config, penalty_weights):
    """
    Computes the objective loss and its exact analytical gradient using
    a provided set of penalty weights.
    """
    N = robot_config['joints_number']
    
    epsilon = penalty_weights['epsilon']
    zeta = penalty_weights['zeta']
    xi = penalty_weights['xi']
    kappa = penalty_weights['kappa']
    eta = penalty_weights['eta'] # New penalty for minimum bends

    intermediates = _compute_fk_and_intermediates(b_vals, robot_config)
    b = intermediates['b']
    x_N = intermediates['coords'][-1]
    control_effort = intermediates['control_effort']

    dist_sq = np.sum(np.square(x_N - goal))
    reg_bin = -epsilon * np.sum(b_vals * (b_vals - 1.0))
    openness = b[0,:] * b[1,:]
    reg_group = zeta * np.sum(np.square(openness[:-1] - openness[1:]))
    reg_bend = xi * np.sum(np.square(control_effort))
    reg_zigzag = 0.0
    if N > 2:
        effort_double_prime = control_effort[2:] - 2 * control_effort[1:-1] + control_effort[:-2]
        reg_zigzag = kappa * np.sum(np.square(effort_double_prime))
    
    # Add the new L1 penalty to encourage sparse bending
    reg_num_bends = eta * np.sum(np.sqrt(control_effort**2 + 1e-9)) # Smooth L1 norm
    
    total_loss = dist_sq + reg_bin + reg_group + reg_bend + reg_zigzag + reg_num_bends

    grad = np.zeros((2, N))
    grad += -epsilon * (2 * b - 1)
    grad_bend_term = 2 * xi * control_effort
    grad[0, :] += grad_bend_term
    grad[1, :] -= grad_bend_term
    if N > 1:
        d_openness = openness[:-1] - openness[1:]
        term = 2 * zeta * d_openness
        grad[0, :-1] += term * b[1, :-1]; grad[0, 1:] -= term * b[1, 1:]
        grad[1, :-1] += term * b[0, :-1]; grad[1, 1:] -= term * b[0, 1:]
    if N > 2:
        term = 2 * kappa * effort_double_prime
        grad_zigzag_term = np.zeros(N)
        grad_zigzag_term[:-2] += term
        grad_zigzag_term[1:-1] -= 2 * term
        grad_zigzag_term[2:] += term
        grad[0, :] += grad_zigzag_term
        grad[1, :] -= grad_zigzag_term
        
    # Add gradient for the L1 penalty
    grad_num_bends_term = eta * control_effort / np.sqrt(control_effort**2 + 1e-9)
    grad[0, :] += grad_num_bends_term
    grad[1, :] -= grad_num_bends_term
        
    err = 2 * (x_N - goal)
    for m in range(N - 1, -1, -1):
        J_col_0, J_col_1 = _compute_end_effector_jacobian_column(m, intermediates)
        grad[0, m] += np.dot(err, J_col_0)
        grad[1, m] += np.dot(err, J_col_1)
        
    return total_loss, grad.flatten()

def _compute_fk_and_intermediates(b_vals, robot_config):
    """A helper to run FK and return all values needed for gradient calcs."""
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
        "b": b, "control_effort": control_effort, "phi": phi, 
        "world_phi": world_phi, "lengths": lengths, "coords": coords,
        "unit_vectors": unit_vectors, "S_beta": S_beta, "delta_L": delta_L
    }

def _compute_point_j_jacobian_column_m(j, m, intermediates):
    """Computes the partial derivative of point j's position w.r.t. actuator m's inputs."""
    if m >= j:
        return np.zeros(2), np.zeros(2)

    p_j = intermediates['coords'][j]
    p_m = intermediates['coords'][m]
    u_m = intermediates['unit_vectors'][m]
    v_m_j = p_j - p_m
    v_m_j_perp = np.array([v_m_j[1], -v_m_j[0]])
    
    denom = np.sqrt(1 - (intermediates['S_beta'] * intermediates['control_effort'][m])**2 + 1e-9)
    dphi_db0 = intermediates['S_beta'] / denom
    dphi_db1 = -intermediates['S_beta'] / denom
    
    dlen_db = intermediates['delta_L'] / 2
    
    J_col_0 = dlen_db * u_m + dphi_db0 * v_m_j_perp
    J_col_1 = dlen_db * u_m + dphi_db1 * v_m_j_perp
    
    return J_col_0, J_col_1

def _compute_end_effector_jacobian_column(m, intermediates):
    """Computes one column of the Jacobian for the end-effector w.r.t. joint m."""
    N = len(intermediates['phi'])
    return _compute_point_j_jacobian_column_m(N, m, intermediates)

# =============================================================================
# Core Inverse Kinematics Logic (Hybrid SLSQP + Targeted Binary Search)
# =============================================================================
def discrete_inverse_kinematics(goal, obstacles_data, robot_config, ik_config):
    """
    Calculates optimal states using a hybrid SLSQP (exploration) and 
    a targeted binary search (refinement) strategy.
    """
    N = robot_config['joints_number']
    max_iter = ik_config['max_iter']
    num_starts = ik_config['num_starts']
    num_critical_joints = ik_config.get('num_critical_joints', 5)

    # --- Define Penalty Weight Set ---
    penalty_weights = {
        'epsilon': (float(ik_config['epsilon_base']) / N),
        'zeta': (float(ik_config['zeta_base']) / N),
        'xi': (float(ik_config['xi_base']) / N),
        'kappa': (float(ik_config['kappa_base']) / N),
        'eta': (float(ik_config['eta_base']) / N) # New penalty
    }

    def init_params():
        return np.clip(np.full(2 * N, 0.5) + (np.random.rand(2 * N) - 0.5) * 0.1, 0, 1)

    # --- Phase 1: Exploration with SciPy SLSQP ---
    print("--- Starting Phase 1: Exploration (SciPy SLSQP) ---")
    best_exploratory_solution, best_exploratory_loss = None, np.inf
    
    loss_func_slsqp = lambda b: calculate_loss_and_analytical_gradient(b, goal, robot_config, penalty_weights)[0]

    bounds = [(0, 1) for _ in range(2 * N)]
    scipy_constraints = []
    # (Scipy constraint setup is unchanged and omitted for brevity)
    for obstacle in obstacles_data:
        if obstacle['type'] == 'circle':
            center = np.array(obstacle['center'])
            radius = obstacle['radius']
            def circle_constraint_func(b_vals, c=center, r=radius):
                all_coords = fk(b_vals, robot_config, return_all_coords=True)
                distances = np.linalg.norm(all_coords - c, axis=1) - r - robot_config['ro']
                return np.min(distances)
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
                distances = np.where(dist_inside < 0, dist_inside, dist_outside) - robot_config['ro']
                return np.min(distances)
            scipy_constraints.append({'type': 'ineq', 'fun': rect_constraint_func})

    optimizer_options = {'maxiter': max_iter, 'ftol': 1e-6, 'disp': False}

    for i in range(num_starts):
        print(f"\n--- Exploration Run {i+1}/{num_starts} ---")
        initial_guess = init_params()
        
        current_result = minimize(loss_func_slsqp, initial_guess, method='SLSQP', 
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

    # --- Phase 2: Targeted Binary Search ---
    print(f"\n--- Starting Phase 2: Targeted Binary Search ---")
    
    # Find the indices of the most non-binary values
    non_binary_scores = np.abs(best_exploratory_solution - 0.5)
    critical_indices = np.argsort(non_binary_scores)[:num_critical_joints]
    
    print(f"Identified {len(critical_indices)} critical actuators to search.")

    # Create a base solution with non-critical values rounded
    base_solution = np.round(best_exploratory_solution)
    
    best_binary_solution = base_solution.copy()
    best_binary_dist = np.linalg.norm(fk(best_binary_solution, robot_config) - goal)

    # Generate all binary combinations for the critical indices
    num_combinations = 2**len(critical_indices)
    print(f"Testing {num_combinations} binary combinations...")

    for i, combination in enumerate(itertools.product([0, 1], repeat=len(critical_indices))):
        print(f"\rTesting combination {i+1}/{num_combinations}", end="")
        
        test_solution = base_solution.copy()
        test_solution[critical_indices] = combination
        
        dist = np.linalg.norm(fk(test_solution, robot_config) - goal)
        
        if dist < best_binary_dist:
            best_binary_dist = dist
            best_binary_solution = test_solution.copy()
            
    print(f"\n--- Phase 2 Complete. Best binary distance: {best_binary_dist:.4f} ---")

    # The final continuous solution is the one from exploration,
    # but the final binary solution is the one from our targeted search.
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
# Visualization (MODIFIED)
# =============================================================================
def plot_results(b_continuous, b_binary, goal, obstacles_data, config, title="Soft Robot Configuration Analysis"):
    """Plots the robot configuration with a customizable title."""
    if b_binary is None: return
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
# Main Execution Block (MODIFIED)
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
    
    GOAL = np.array(env_config.get('goal', [-400, 300]))
    OBSTACLES = env_config.get('obstacles', [])

    action_sequence, res_bits, res_continuous = discrete_inverse_kinematics(GOAL, OBSTACLES, robot_config, ik_config)

    if action_sequence is not None:
        print("\n" + "="*30)
        print("      Final Results")
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
