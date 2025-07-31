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
# Multiplier for the continuous "hyper-robot". It will have N * CONTINUOUS_DOF_MULTIPLIER joints.
CONTINUOUS_DOF_MULTIPLIER = 2 

# FABRIK solver settings
NUM_FABRIK_ITERATIONS = 30 # Increased iterations for better convergence with obstacles.

# Spline settings
SPLINE_SMOOTHING_FACTOR = 500.0 

# SA Spline Fitting settings
# Increased the weight for the end-effector error to strongly prioritize reaching the goal.
W_EE_ERROR = 5.0 

# =============================================================================
# Forward Kinematics & Core Math
# =============================================================================
def fk(b_vals: np.ndarray, robot_config: Dict[str, Any], return_all_coords: bool = False) -> np.ndarray:
    """
    Computes the forward kinematics. This version is general for continuous or discrete inputs.
    """
    N = robot_config['joints_number']
    folded_len = robot_config['folded_cell_length']
    deployed_len = robot_config['deployed_cell_length']
    bend_angle_const = robot_config['bend_angle_const']
    
    b = b_vals.reshape((2, N))
    
    phi = (b[0,:] - b[1,:]) * bend_angle_const

    initial_angle = np.pi / 2
    world_phi = np.cumsum(phi) + initial_angle
    
    b_avg = (b[0,:] + b[1,:]) / 2
    lengths = folded_len + (deployed_len - folded_len) * b_avg
    
    coords = np.zeros((N + 1, 2))
    unit_vectors = np.array([np.cos(world_phi), np.sin(world_phi)])
    
    for i in range(N):
        coords[i+1, :] = coords[i, :] + lengths[i] * unit_vectors[:, i]
        
    return coords if return_all_coords else coords[-1]

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

def project_out_of_obstacles(point: np.ndarray, obstacles: List[Dict[str, Any]]) -> np.ndarray:
    """
    If a point is inside any obstacle, project it to the nearest point on its surface.
    """
    for obs in obstacles:
        if is_in_obstacle(point, obs):
            obs_type = obs.get('type')
            if obs_type == 'rectangle':
                v = np.array(obs['vertices'])
                # Find the closest point on the rectangle's boundary
                px = np.clip(point[0], v[0,0], v[1,0])
                py = np.clip(point[1], v[0,1], v[1,1])
                # Determine which edge is closest
                dist_to_left = abs(point[0] - v[0,0])
                dist_to_right = abs(point[0] - v[1,0])
                dist_to_bottom = abs(point[1] - v[0,1])
                dist_to_top = abs(point[1] - v[1,1])
                min_dist = min(dist_to_left, dist_to_right, dist_to_bottom, dist_to_top)
                if min_dist == dist_to_left:
                    return np.array([v[0,0], point[1]])
                elif min_dist == dist_to_right:
                    return np.array([v[1,0], point[1]])
                elif min_dist == dist_to_bottom:
                    return np.array([point[0], v[0,1]])
                else: # dist_to_top
                    return np.array([point[0], v[1,1]])
            elif obs_type == 'circle':
                center = np.array(obs['center'])
                radius = obs['radius']
                vec_from_center = point - center
                dist_from_center = np.linalg.norm(vec_from_center)
                if dist_from_center < 1e-6: # Point is at the center
                    return center + np.array([radius, 0])
                return center + (vec_from_center / dist_from_center) * radius
    return point


# =============================================================================
# Stage 1: Continuous Pathfinding with FABRIK
# =============================================================================
def solve_continuous_with_fabrik(goal, obstacles, robot_config, ik_config):
    """
    Solves for a smooth, obstacle-avoiding path using a FABRIK-style iterative solver.
    """
    print("\n--- Stage 1: Solving for continuous path with FABRIK ---")
    
    continuous_config = robot_config.copy()
    original_N = robot_config['joints_number']
    continuous_N = original_N * CONTINUOUS_DOF_MULTIPLIER
    continuous_config['joints_number'] = continuous_N

    start_pos = np.array([0.0, 0.0])
    
    total_length_guess = np.linalg.norm(goal - start_pos)
    segment_lengths = np.full(continuous_N, total_length_guess / continuous_N if continuous_N > 0 else 0)
    
    P_joints = np.zeros((continuous_N + 1, 2))
    direction_to_goal = (goal - start_pos) / total_length_guess if total_length_guess > 0 else np.array([0,1])
    for i in range(1, continuous_N + 1):
        P_joints[i] = P_joints[i-1] + direction_to_goal * segment_lengths[i-1]

    # Main FABRIK iteration loop
    for iter_num in range(NUM_FABRIK_ITERATIONS):
        print(f"FABRIK Iteration: {iter_num + 1}/{NUM_FABRIK_ITERATIONS}", end='\r')
        
        # Backward pass
        P_joints[-1] = goal
        for i in range(continuous_N - 1, -1, -1):
            direction = P_joints[i] - P_joints[i+1]
            dist = np.linalg.norm(direction)
            if dist > 1e-6:
                P_joints[i] = P_joints[i+1] + (direction / dist) * segment_lengths[i]
            # Project out of obstacles after moving
            P_joints[i] = project_out_of_obstacles(P_joints[i], obstacles)

        # Forward pass
        P_joints[0] = start_pos
        for i in range(continuous_N):
            direction = P_joints[i+1] - P_joints[i]
            dist = np.linalg.norm(direction)
            if dist > 1e-6:
                P_joints[i+1] = P_joints[i] + (direction / dist) * segment_lengths[i]
            # Project out of obstacles after moving
            P_joints[i+1] = project_out_of_obstacles(P_joints[i+1], obstacles)

    print("\nFABRIK iterations finished.")
    
    if check_collision(P_joints, obstacles):
        print("Warning: Continuous path generated by FABRIK still has a collision.")

    print("Stage 1 finished.")
    return P_joints

# =============================================================================
# Stage 3: SA to Fit Discrete Robot to Spline
# =============================================================================
def fit_discrete_robot_to_spline(spline_tck, goal, robot_config, obstacles):
    """
    Uses SA to find the best binary actuator states to match the target spline.
    """
    print("\n--- Stage 3: Fitting discrete robot to spline with SA ---")
    N = robot_config['joints_number']

    def cost_to_spline(b_candidate: np.ndarray) -> float:
        points_candidate = fk(b_candidate, robot_config, return_all_coords=True)
        if check_collision(points_candidate, obstacles):
            return 1e9
            
        u_eval = np.linspace(0, 1, N + 1)
        points_spline_target = np.array(splev(u_eval, spline_tck)).T
        shape_cost = np.mean(np.linalg.norm(points_candidate - points_spline_target, axis=1)**2)
        
        end_effector_pos = points_candidate[-1]
        goal_cost = np.linalg.norm(end_effector_pos - goal)**2
        
        return shape_cost + W_EE_ERROR * goal_cost

    sa_params = {
        'initial_temp': 10000.0,
        'final_temp': 100.0,
        'cooling_rate': 0.996,
        'max_iter_per_temp': 200
    }
    
    current_b = np.random.randint(0, 2, 2 * N)
    current_cost = cost_to_spline(current_b)
    best_b, best_cost = np.copy(current_b), current_cost
    temperature = sa_params['initial_temp']

    while temperature > sa_params['final_temp']:
        print(f"Temp: {temperature:.2f}, Best Cost: {best_cost:.4f}", end='\r')
        for _ in range(sa_params['max_iter_per_temp']):
            neighbor_b = np.copy(current_b)
            flip_index = random.randint(0, len(neighbor_b) - 1)
            neighbor_b[flip_index] = 1 - neighbor_b[flip_index]
            
            neighbor_cost = cost_to_spline(neighbor_b)
            cost_diff = neighbor_cost - current_cost
            
            if cost_diff < 0 or random.random() < math.exp(-cost_diff / temperature):
                current_b, current_cost = neighbor_b, neighbor_cost
            
            if current_cost < best_cost:
                best_b, best_cost = np.copy(current_b), current_cost
        
        temperature *= sa_params['cooling_rate']
        
    print("\nStage 3 finished.")
    return best_b

# =============================================================================
# Visualization
# =============================================================================
def plot_results(continuous_path, final_b, spline_tck, g, obstacles, robot_config, env_config):
    """
    Visualizes all stages of the hybrid solution process.
    """
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 12))
    
    points_final = fk(final_b, robot_config, return_all_coords=True)
    
    u_fine = np.linspace(0, 1, 500)
    spline_points = splev(u_fine, spline_tck)
    ax1.plot(spline_points[0], spline_points[1], ':', color='orange', linewidth=2.5, label='Target Spline', zorder=2)
    
    ax1.plot(continuous_path[:, 0], continuous_path[:, 1], 'o--', label='Continuous Path', color='gray', linewidth=2, markersize=3, alpha=0.7, zorder=3)
    
    for obs in obstacles:
        obs_type = obs.get('type')
        if obs_type == 'rectangle':
            v = np.array(obs['vertices'])
            rect = patches.Rectangle(v[0], v[1,0]-v[0,0], v[1,1]-v[0,1], linewidth=1, edgecolor='r', facecolor='salmon', alpha=0.5, label='Obstacle', zorder=1)
            ax1.add_patch(rect)

    ax1.plot(points_final[:, 0], points_final[:, 1], 'o-', label='Final Discrete Shape', color='royalblue', linewidth=2.5, markersize=4, zorder=4)
    
    ax1.plot(0, 0, 'ks', markersize=10, label='Base', zorder=5)
    ax1.plot(g[0], g[1], 'g*', markersize=15, label='Goal', zorder=5)
    ax1.plot(points_final[-1, 0], points_final[-1, 1], 'ro', markersize=10, label='Final End-Effector', zorder=5)
    
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
    
    GOAL = np.array(env_config.get('goal', [-200, 600]))
    OBSTACLES = env_config.get('obstacles', [])
    START = np.array([0.0, 0.0])

    # --- Stage 1: Find a smooth path with a continuous, high-DOF robot using FABRIK ---
    continuous_path_points = solve_continuous_with_fabrik(GOAL, OBSTACLES, robot_config, ik_config)

    # --- Stage 2: Create a smooth spline from the continuous path ---
    print("\n--- Stage 2: Creating a smooth spline from the continuous path ---")
    tck, u = splprep([continuous_path_points[:, 0], continuous_path_points[:, 1]], s=SPLINE_SMOOTHING_FACTOR, k=min(3, len(continuous_path_points)-1))
    print("Stage 2 finished.")

    # --- Stage 3: Fit the discrete robot to the spline using SA ---
    final_b = fit_discrete_robot_to_spline(tck, GOAL, robot_config, OBSTACLES)

    # --- Final Analysis ---
    print("\n" + "="*40)
    print("           Final Results Analysis")
    print("="*40)

    points_final = fk(final_b, robot_config, return_all_coords=True)
    pos_final = points_final[-1]
    err_final = np.linalg.norm(pos_final - GOAL)
    coll_final = check_collision(points_final, OBSTACLES)

    print(f"Target Goal (X, Z):        {GOAL}")
    print(f"Final Position (X, Z):     {np.round(pos_final, 4)}")
    print(f"Final Distance Err:        {err_final:.6f}")
    print(f"Collision in final state:  {coll_final}")
    print("="*40)
    
    plot_results(continuous_path_points, final_b, tck, GOAL, OBSTACLES, robot_config, env_config)