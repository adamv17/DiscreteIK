import numpy as np
import math
import random
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Dict, Any
from scipy.interpolate import splprep, splev

# =============================================================================
# Configuration
# =============================================================================
# This parameter controls the smoothing of the target spline.
# A smaller value means the spline will follow the original points more closely.
# A larger value will create a smoother, more generalized curve.
# A good starting point is the number of joints.
SPLINE_SMOOTHING_FACTOR = 150.0

# =============================================================================
# Forward Kinematics & Core Math
# =============================================================================
def fk(b_vals: np.ndarray, robot_config: Dict[str, Any], return_all_coords: bool = False) -> np.ndarray:
    """
    Computes the forward kinematics for the robot using parameters from config.
    The robot operates in the X-Z plane, with Z being the vertical axis.
    """
    N = robot_config['joints_number']
    folded_len = robot_config['folded_cell_length']
    deployed_len = robot_config['deployed_cell_length']
    bend_angle = robot_config['bend_angle_const']
    
    b = b_vals.reshape((2, N))
    phi = (b[0,:] - b[1,:]) * bend_angle

    initial_angle = np.pi / 2
    world_phi = np.cumsum(phi) + initial_angle
    
    b_avg = (b[0,:] + b[1,:]) / 2
    lengths = folded_len + (deployed_len - folded_len) * b_avg
    
    coords = np.zeros((N + 1, 2))
    unit_vectors = np.array([np.cos(world_phi), np.sin(world_phi)])
    
    for i in range(N):
        coords[i+1, :] = coords[i, :] + lengths[i] * unit_vectors[:, i]
        
    if return_all_coords:
        return coords
    return coords[-1]

# =============================================================================
# Obstacle Collision Checking
# =============================================================================
def check_collision(points: np.ndarray, obstacles: List[Dict[str, Any]]) -> bool:
    """Checks if any point in the robot's body collides with any obstacle."""
    if not obstacles: return False
    for p in points:
        for obs in obstacles:
            if is_in_obstacle(p, obs): return True
    return False

def is_in_obstacle(p: np.ndarray, obstacle: Dict[str, Any]) -> bool:
    """Checks if a point p (x, z) is inside a given obstacle."""
    obs_type = obstacle.get('type')
    if obs_type == 'rectangle':
        v = np.array(obstacle['vertices'])
        return (p[0] >= v[0,0] and p[0] <= v[1,0] and p[1] >= v[0,1] and p[1] <= v[1,1])
    elif obs_type == 'circle':
        center = np.array(obstacle['center'])
        radius = obstacle['radius']
        return np.sum((p - center)**2) < radius**2
    return False

# =============================================================================
# Cost Function (for initial IK)
# =============================================================================
def cost_function_ik(b: np.ndarray, goal: np.ndarray, obstacles: List[Dict], robot_config: Dict, ik_config: Dict) -> float:
    """
    Calculates the cost for the initial inverse kinematics problem.
    """
    N = robot_config['joints_number']
    
    all_points = fk(b, robot_config, return_all_coords=True)
    if check_collision(all_points, obstacles):
        return 1e9 

    final_pos = all_points[-1]
    b_reshaped = b.reshape((2, N))

    cost = np.sum((final_pos - goal)**2)

    kappa = float(ik_config['kappa_base']) *100 / N
    bend_diff = b_reshaped[0,:] - b_reshaped[1,:]
    smoothness_penalty = kappa * np.sum(np.diff(bend_diff, 2)**2)
    cost += smoothness_penalty

    eta = float(ik_config['eta_base'])
    b1, b2 = b_reshaped[0,:], b_reshaped[1,:]
    bending_penalty = eta * sum(b1 != b2)
    cost += bending_penalty
    
    return cost

# =============================================================================
# Simulated Annealing Solver
# =============================================================================
def simulated_annealing(cost_func, b_initial, cost_func_args, sa_params) -> Tuple[np.ndarray, float, List[float]]:
    """
    A generalized Simulated Annealing solver.
    """
    N = len(b_initial) // 2
    
    current_b = np.copy(b_initial)
    current_cost = cost_func(current_b, *cost_func_args)
    
    best_b = np.copy(current_b)
    best_cost = current_cost
    
    temperature = sa_params['initial_temp']
    cost_history = [best_cost]

    while temperature > sa_params['final_temp']:
        print(f"Temp: {temperature:.2f}, Best Cost: {best_cost:.4f}", end='\r')
        for _ in range(sa_params['max_iter_per_temp']):
            neighbor_b = np.copy(current_b)
            flip_index = random.randint(0, len(neighbor_b) - 1)
            neighbor_b[flip_index] = 1 - neighbor_b[flip_index]
            
            neighbor_cost = cost_func(neighbor_b, *cost_func_args)
            cost_diff = neighbor_cost - current_cost
            
            if cost_diff < 0 or random.random() < math.exp(-cost_diff / temperature):
                current_b, current_cost = neighbor_b, neighbor_cost
            
            if current_cost < best_cost:
                best_b, best_cost = np.copy(current_b), current_cost
        
        temperature *= sa_params['cooling_rate']
        cost_history.append(best_cost)
        
    print() # Newline after finishing
    return best_b, best_cost, cost_history

# =============================================================================
# Post-Processing with Spline Fitting and Re-Optimization
# =============================================================================
def post_process_with_spline(b_original: np.ndarray, goal: np.ndarray, obstacles: List[Dict], robot_config: Dict) -> Tuple[np.ndarray, Any]:
    """
    Refines the robot's shape by fitting a spline that incorporates the goal.
    """
    print("\n--- Starting Spline-Based Post-Processing ---")
    N = robot_config['joints_number']

    # --- Stage 1: Fit a smooth spline to the original solution, forced to the goal ---
    print("Step 1: Fitting a goal-aware B-spline to the initial solution...")
    points_orig = fk(b_original, robot_config, return_all_coords=True)
    
    # Create a new set of points for spline fitting.
    # This set is the original solution's points, but with the end-effector
    # position replaced by the actual goal. This forces the spline to be accurate.
    points_for_spline = np.copy(points_orig)
    points_for_spline[-1] = goal
    
    # splprep finds the B-spline representation of a curve.
    # s is a smoothing factor. A higher s means a smoother curve.
    tck, u = splprep([points_for_spline[:, 0], points_for_spline[:, 1]], s=SPLINE_SMOOTHING_FACTOR, k=3)
    
    # --- Stage 2: Re-optimize to fit the robot to the new spline ---
    print("Step 2: Re-optimizing the robot configuration to match the spline...")

    def cost_to_spline(b_candidate: np.ndarray, spline_tck: Any, obstacles: List[Dict], robot_config: Dict) -> float:
        """
        Cost function that penalizes deviation from a target spline.
        """
        points_candidate = fk(b_candidate, robot_config, return_all_coords=True)
        
        if check_collision(points_candidate, obstacles):
            return 1e9

        # Sample points on the spline that correspond to each joint
        u_eval = np.linspace(0, 1, N + 1)
        points_spline_target = np.array(splev(u_eval, spline_tck)).T
        
        # Calculate the sum of squared distances between the robot's joints and the spline
        dist_cost = np.sum(np.linalg.norm(points_candidate - points_spline_target, axis=1)**2)
        
        return dist_cost

    # Use a second, faster SA run for this refinement step
    sa_params_refine = {
        'initial_temp': 2000.0,
        'final_temp': 1.0,
        'cooling_rate': 0.99,
        'max_iter_per_temp': 100
    }
    
    # Start the refinement from the original solution
    b_processed, _, _ = simulated_annealing(
        cost_to_spline,
        b_original,
        (tck, obstacles, robot_config),
        sa_params_refine
    )
    
    print("--- Post-Processing Finished ---")
    return b_processed, tck

# =============================================================================
# Visualization
# =============================================================================
def plot_results(b_original: np.ndarray, b_processed: np.ndarray, spline_tck: Any, g: np.ndarray, cost_history: List[float], obstacles: List[Dict], robot_config: Dict, env_config: Dict):
    """
    Visualizes the final robot configuration, cost, and environment.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
    
    # --- Plot 1: Final Robot Configuration Comparison ---
    points_orig = fk(b_original, robot_config, return_all_coords=True)
    points_proc = fk(b_processed, robot_config, return_all_coords=True)
    
    # Plot the target spline
    u_fine = np.linspace(0, 1, 500)
    spline_points = splev(u_fine, spline_tck)
    ax1.plot(spline_points[0], spline_points[1], ':', color='orange', linewidth=2.5, label='Target Spline', zorder=2)
    
    for obs in obstacles:
        obs_type = obs.get('type')
        if obs_type == 'rectangle':
            v = np.array(obs['vertices'])
            rect = patches.Rectangle(v[0], v[1,0]-v[0,0], v[1,1]-v[0,1], linewidth=1, edgecolor='r', facecolor='salmon', alpha=0.5, label='Obstacle', zorder=1)
            ax1.add_patch(rect)

    ax1.plot(points_orig[:, 0], points_orig[:, 1], 'o--', label='Original Solution', color='gray', linewidth=2, markersize=3, alpha=0.7, zorder=3)
    ax1.plot(points_proc[:, 0], points_proc[:, 1], 'o-', label='Post-Processed', color='royalblue', linewidth=2.5, markersize=4, zorder=4)
    
    ax1.plot(0, 0, 'ks', markersize=10, label='Base', zorder=5)
    ax1.plot(g[0], g[1], 'g*', markersize=15, label='Original Target', zorder=5)
    ax1.plot(points_proc[-1, 0], points_proc[-1, 1], 'ro', markersize=10, label='Final End-Effector', zorder=5)
    
    ax1.set_title(f"Final Robot Configuration (N={robot_config['joints_number']})", fontsize=14)
    ax1.set_xlabel('X Position (mm)', fontsize=12)
    ax1.set_ylabel('Z Position (mm)', fontsize=12)
    
    bounds = env_config.get('bounds', {'x': [-500, 500], 'z': [0, 1200]})
    ax1.set_xlim(bounds['x'])
    ax1.set_ylim(bounds['z'])
    
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), fontsize=10)
    
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_aspect('equal', adjustable='box')

    # --- Plot 2: Cost Convergence ---
    ax2.plot(cost_history, color='orangered', linewidth=2)
    ax2.set_title('Initial IK Cost Convergence', fontsize=14)
    ax2.set_xlabel('Temperature Steps', fontsize=12)
    ax2.set_ylabel('Best Cost Found (log scale)', fontsize=12)
    ax2.set_yscale('log')
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout(pad=3.0)
    plt.show()

# =============================================================================
# Main Execution Block
# =============================================================================
if __name__ == "__main__":
    try:
        with open('robot_config.yaml', 'r') as file: robot_config = yaml.safe_load(file)
        with open('env.yaml', 'r') as file: env_config = yaml.safe_load(file)
        with open('ik_config.yaml', 'r') as file: ik_config = yaml.safe_load(file)
    except FileNotFoundError as e:
        print(f"Error: Config file not found. {e}")
        exit()
    
    GOAL = np.array(env_config.get('goal', [0, 800]))
    OBSTACLES = env_config.get('obstacles', [])

    print("--- Starting Initial Inverse Kinematics Search ---")
    
    sa_params_ik = {
        'initial_temp': 5000.0,
        'final_temp': 5.0,
        'cooling_rate': 0.996,
        'max_iter_per_temp': 200
    }
    
    initial_b = np.random.randint(0, 2, 2 * robot_config['joints_number'])
    
    optimal_b, final_cost, history = simulated_annealing(
        cost_function_ik,
        initial_b,
        (GOAL, OBSTACLES, robot_config, ik_config),
        sa_params_ik
    )
    
    processed_b, spline_tck = post_process_with_spline(optimal_b, GOAL, OBSTACLES, robot_config)

    print("\n" + "="*40)
    print("           Final Results Analysis")
    print("="*40)

    points_orig = fk(optimal_b, robot_config, return_all_coords=True)
    pos_orig = points_orig[-1]
    err_orig = np.linalg.norm(pos_orig - GOAL)
    coll_orig = check_collision(points_orig, OBSTACLES)
    print("\n--- Original SA Solution ---")
    print(f"Final Position (X, Z):     {np.round(pos_orig, 4)}")
    print(f"Final Distance Err:        {err_orig:.6f}")
    print(f"Collision in final state:  {coll_orig}")

    points_proc = fk(processed_b, robot_config, return_all_coords=True)
    pos_proc = points_proc[-1]
    err_proc = np.linalg.norm(pos_proc - GOAL)
    coll_proc = check_collision(points_proc, OBSTACLES)
    print("\n--- Post-Processed Solution ---")
    print(f"Final Position (X, Z):     {np.round(pos_proc, 4)}")
    print(f"Final Distance Err:        {err_proc:.6f}")
    print(f"Collision in final state:  {coll_proc}")
    if coll_proc and not coll_orig:
        print("WARNING: Post-processing may have introduced a collision!")
    print("="*40)
    
    plot_results(optimal_b, processed_b, spline_tck, GOAL, history, OBSTACLES, robot_config, env_config)
