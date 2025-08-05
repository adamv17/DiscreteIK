import numpy as np
import math
import random
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Dict, Any
from scipy.optimize import minimize
import itertools

# =============================================================================
# Configuration
# =============================================================================
# Number of pieces/segments to use in the initial continuous optimization.
# Note: The number of angle combinations to check is 3^NUM_PIECES.
# Keep this value small (e.g., 3-5) for reasonable performance.
NUM_PIECES = 5 

# New penalty for the number of bends in the ideal path.
# A larger value will result in straighter paths.
BEND_NUMBER_PENALTY = 2000.0

# =============================================================================
# Forward Kinematics & Core Math
# =============================================================================
def fk(b_vals: np.ndarray, robot_config: Dict[str, Any], return_all_coords: bool = False) -> np.ndarray:
    """
    Computes the forward kinematics from a binary actuator vector.
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
        return (p[0] >= min(v[0,0],v[1,0]) and p[0] <= max(v[0,0],v[1,0]) and p[1] >= min(v[0,1],v[1,1]) and p[1] <= max(v[0,1],v[1,1]))
    elif obs_type == 'circle':
        center = np.array(obstacle['center'])
        radius = obstacle['radius']
        return np.sum((p - center)**2) < radius**2
    return False

# =============================================================================
# Two-Stage "Plan-then-Fit" Solver
# =============================================================================
def solve_with_plan_and_fit(goal, obstacles, robot_config, ik_config):
    """
    Solves the IK problem using a two-stage approach:
    1. A continuous optimization finds an ideal geometric path with a fixed number of bends.
    2. A deterministic forward construction fits the real robot to this path.
    """
    N = robot_config['joints_number']
    
    # --- Stage 1: Find an ideal geometric path via continuous optimization ---
    print("\n--- Stage 1: Finding Ideal Geometric Path with Continuous Optimization ---")
    
    bend_angle = robot_config['bend_angle_const']

    def geometric_path_from_continuous(lengths, angles):
        """A simplified FK for the continuous path planning stage."""
        path_points = [np.array([0.0, 0.0])]
        cumulative_angle = np.pi / 2
        for i in range(NUM_PIECES):
            cumulative_angle += angles[i]
            new_point = path_points[-1] + lengths[i] * np.array([math.cos(cumulative_angle), math.sin(cumulative_angle)])
            path_points.append(new_point)
        return np.array(path_points)

    def cost_and_grad_func(lengths, angles):
        """Cost function for the continuous path planning stage.
           This function now ONLY returns position error or collision penalty.
        """
        path = geometric_path_from_continuous(lengths, angles)
        
        # Create a denser path for a more robust collision check
        dense_path_points = [path[0]]
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i+1]
            segment_len = np.linalg.norm(p2 - p1)
            # Add an intermediate point every 5mm for collision checking
            num_intermediate_points = int(segment_len / 1.0)
            if num_intermediate_points > 1:
                for j in range(1, num_intermediate_points):
                    interp_point = p1 + (p2 - p1) * (j / num_intermediate_points)
                    dense_path_points.append(interp_point)
            dense_path_points.append(p2)
        
        if check_collision(np.array(dense_path_points), obstacles):
            return 1e9, np.zeros_like(lengths)

        final_pos = path[-1]
        error_vec = final_pos - goal
        cost = np.linalg.norm(error_vec)**2
    
        # Analytical Gradient Calculation
        grad = np.zeros_like(lengths)
        # The jacobian of the final position w.r.t. the lengths is just the unit vectors of each piece
        cumulative_angle = np.pi / 2
        for k in range(NUM_PIECES):
            cumulative_angle += angles[k]
            unit_vec_k = np.array([math.cos(cumulative_angle), math.sin(cumulative_angle)])
            grad[k] = 2 * np.dot(error_vec, unit_vec_k)
                
        return cost, grad

    # Exhaustively check all angle combinations
    possible_angles = [-2*bend_angle, 0, 2*bend_angle]
    angle_combinations = list(itertools.product(possible_angles, repeat=NUM_PIECES))
    
    best_result = {'score': float('inf'), 'lengths': None, 'angles': None}

    for i, angles in enumerate(angle_combinations):
        print(f"Testing angle combination {i+1}/{len(angle_combinations)}", end='\r')
        
        l0 = [100, 100, 100, 100, 100]
        bounds = [(0, robot_config['deployed_cell_length'] * N)] * NUM_PIECES
        
        res = minimize(lambda l: cost_and_grad_func(l, angles), l0,  method='L-BFGS-B', jac=True, bounds=bounds)
        
        # Calculate a total score that includes the bend penalty
        position_or_collision_cost = res.fun
        num_bends = np.count_nonzero(angles)
        bend_penalty = BEND_NUMBER_PENALTY * num_bends
        total_score = position_or_collision_cost + bend_penalty

        if total_score < best_result['score']:
            best_result['score'] = total_score
            best_result['lengths'] = res.x
            best_result['angles'] = angles
    
    print("\n%d" % best_result['score'])
    print("\nStage 1 (Continuous Optimization) finished.")
    
    ideal_path_points_coarse = geometric_path_from_continuous(best_result['lengths'], best_result['angles'])

    # --- Stage 2: Fit the real robot to the ideal geometric path ---
    print("\n--- Stage 2: Fitting Robot to Ideal Path via Length Optimization ---")
    
    b = np.zeros((2,N))
    total_ideal_length = sum(best_result['lengths'])
    
    if total_ideal_length < 1e-6:
        print("Warning: Ideal path has zero length.")
        return b, ideal_path_points_coarse

    # Allocate total frustums proportionally to the length of each ideal piece
    frustum_allocations = np.round((np.array(best_result['lengths']) / total_ideal_length) * N).astype(int)
    print(frustum_allocations)
    print(best_result['angles'])
    allocation_diff = N - sum(frustum_allocations)
    if allocation_diff != 0 and len(frustum_allocations) > 0:
        frustum_allocations[np.argmax(frustum_allocations)] += allocation_diff

    # Build the final binary actuator vector
    current_frustum_idx = 0
    folded_len = robot_config['folded_cell_length']
    deployed_len = robot_config['deployed_cell_length']
    len_diff = deployed_len - folded_len

    for i in range(NUM_PIECES):
        num_frustums_for_piece = frustum_allocations[i]
        angle_for_piece = best_result['angles'][i]
        ideal_len_for_piece = best_result['lengths'][i]

        end_idx = current_frustum_idx + num_frustums_for_piece
        M = int(abs(angle_for_piece / bend_angle))
        
        if angle_for_piece != 0: # It's a bent piece
            state = [0, 1] if angle_for_piece < 0 else [1, 0]
            for m in range(M):
                if current_frustum_idx < N: 
                    b[0, current_frustum_idx] = state[0]
                    b[1, current_frustum_idx] = state[1]
                    current_frustum_idx += 1
        
        # It's a straight piece, solve for deployment
        num_straight_frustums = num_frustums_for_piece - M
        if num_straight_frustums > 0:
            if len_diff > 1e-6:
                n_deployed_float = (ideal_len_for_piece - num_straight_frustums * folded_len) / len_diff
                n_deployed = int(np.round(np.clip(n_deployed_float, 0, num_straight_frustums)))
            else:
                n_deployed = 0 # Cannot deploy if lengths are the same
            
            n_folded = num_straight_frustums - n_deployed

            # Assign states for this piece
            for j in range(current_frustum_idx, current_frustum_idx + n_deployed):
                if j < N: # Deployed
                    b[0,j] = 1 
                    b[1,j] = 1
            for j in range(current_frustum_idx + n_deployed, end_idx):
                if j < N: # Folded
                    b[0,j] = 0
                    b[1,j] = 0

        current_frustum_idx = end_idx

    print("\nStage 2 (Robot Fitting) finished.")
    return b.flatten(), ideal_path_points_coarse


# =============================================================================
# Visualization
# =============================================================================
def plot_results(final_b, ideal_path, g, obstacles, robot_config, env_config):
    """
    Visualizes the final robot configuration and the ideal path.
    """
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 12))
    points_final = fk(final_b, robot_config, return_all_coords=True)
    ax1.plot(ideal_path[:, 0], ideal_path[:, 1], 'o--', label='Ideal Geometric Path', color='gray', linewidth=2, markersize=3, alpha=0.7, zorder=3)
    for obs in obstacles:
        obs_type = obs.get('type')
        if obs_type == 'rectangle':
            v = np.array(obs['vertices'])
            rect = patches.Rectangle(v[0], v[1,0]-v[0,0], v[1,1]-v[0,1], linewidth=1, edgecolor='r', facecolor='salmon', alpha=0.5, label='Obstacle', zorder=1)
            ax1.add_patch(rect)
    ax1.plot(points_final[:, 0], points_final[:, 1], 'o-', label='Final Robot Shape', color='royalblue', linewidth=2.5, markersize=4, zorder=4)
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
    
    GOAL = np.array(env_config.get('goal', [-300, 400]))
    OBSTACLES = env_config.get('obstacles', [])
    
    final_b, ideal_path = solve_with_plan_and_fit(GOAL, OBSTACLES, robot_config, ik_config)

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
    
    plot_results(final_b, ideal_path, GOAL, OBSTACLES, robot_config, env_config)
