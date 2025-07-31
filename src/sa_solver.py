import numpy as np
import math
import random
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Dict, Any

# =============================================================================
# Forward Kinematics & Core Math
# =============================================================================
def fk(b_vals: np.ndarray, robot_config: Dict[str, Any], return_all_coords: bool = False) -> np.ndarray:
    """
    Computes the forward kinematics for the robot using parameters from config.
    The robot operates in the X-Z plane, with Z being the vertical axis.
    """
    N = robot_config['joints_number']
    D = 2 * robot_config['ro']
    folded_len = robot_config['folded_cell_length']
    deployed_len = robot_config['deployed_cell_length']
    bend_angle = robot_config['bend_angle_const']
    
    b = b_vals.reshape((2, N))
    
    # For binary inputs, the logic simplifies. 
    # (b1-b2) can be -1, 0, or 1, resulting in a bend of -bend_angle, 0, or +bend_angle.
    phi = (b[0,:] - b[1,:]) * bend_angle

    # Cumulative angle, starting with an initial angle of pi/2 to point "up" along Z.
    initial_angle = np.pi / 2
    world_phi = np.cumsum(phi) + initial_angle
    
    b_avg = (b[0,:] + b[1,:]) / 2
    lengths = folded_len + (deployed_len - folded_len) * b_avg
    
    coords = np.zeros((N + 1, 2))
    # Unit vectors for an angle measured from the positive X-axis.
    # The second component corresponds to the Z coordinate.
    unit_vectors = np.array([np.cos(world_phi), np.sin(world_phi)])
    
    for i in range(N):
        coords[i+1, :] = coords[i, :] + lengths[i] * unit_vectors[:, i]
        
    if return_all_coords:
        return coords
    return coords[-1]

# =============================================================================
# Obstacle Collision Checking
# =============================================================================
def is_in_obstacle(p: np.ndarray, obstacle: Dict[str, Any]) -> bool:
    """Checks if a point p (x, z) is inside a given obstacle."""
    obs_type = obstacle.get('type')
    if obs_type == 'rectangle':
        v = np.array(obstacle['vertices'])
        # Assumes vertices are [bottom_left, top_right]
        return (p[0] >= v[0,0] and p[0] <= v[1,0] and p[1] >= v[0,1] and p[1] <= v[1,1])
    elif obs_type == 'circle':
        center = np.array(obstacle['center'])
        radius = obstacle['radius']
        return np.sum((p - center)**2) < radius**2
    return False

def check_collision(points: np.ndarray, obstacles: List[Dict[str, Any]]) -> bool:
    """Checks if any point in the robot's body collides with any obstacle."""
    if not obstacles:
        return False
    for p in points:
        for obs in obstacles:
            if is_in_obstacle(p, obs):
                return True
    return False

# =============================================================================
# Cost Function
# =============================================================================
def cost_function(b: np.ndarray, goal: np.ndarray, obstacles: List[Dict], robot_config: Dict, ik_config: Dict) -> float:
    """
    Calculates the cost of a given robot configuration based on configs.
    """
    N = robot_config['joints_number']
    
    # --- Obstacle Collision Penalty ---
    all_points = fk(b, robot_config, return_all_coords=True)
    if check_collision(all_points, obstacles):
        return 1e9  # Return a massive penalty for any collision

    # --- Cost Calculation (if no collision) ---
    final_pos = all_points[-1]
    b_reshaped = b.reshape((2, N))

    # 1. Position error
    cost = np.sum((final_pos - goal)**2)

    # 2. Smoothness regularizer
    kappa = float(ik_config['kappa_base']) / N
    bend_diff = b_reshaped[0,:] - b_reshaped[1,:]
    smoothness_penalty = kappa * np.sum(np.diff(bend_diff, 2)**2)
    cost += smoothness_penalty

    # 3. Bending penalty
    eta = float(ik_config['eta_base'])
    b1, b2 = b_reshaped[0,:], b_reshaped[1,:]
    bending_penalty = eta*sum(b1 != b2)
    cost += bending_penalty
    
    return cost

# =============================================================================
# Simulated Annealing Solver
# =============================================================================
def simulated_annealing(target_goal: np.ndarray, obstacles: List[Dict], robot_config: Dict, ik_config: Dict) -> Tuple[np.ndarray, float, List[float]]:
    """
    Performs Simulated Annealing to find the optimal discrete actuator state.
    """
    print("Starting Simulated Annealing with loaded configurations...")
    
    N = robot_config['joints_number']
    
    # SA Parameters
    INITIAL_TEMPERATURE = 5000.0
    FINAL_TEMPERATURE = 5.0
    COOLING_RATE = 0.996
    MAX_ITERATIONS_PER_TEMP = 200

    # Initialization
    current_b = np.random.randint(0, 2, 2 * N)
    current_cost = cost_function(current_b, target_goal, obstacles, robot_config, ik_config)
    
    best_b = np.copy(current_b)
    best_cost = current_cost
    
    temperature = INITIAL_TEMPERATURE
    cost_history = [best_cost]

    # Main Annealing Loop
    while temperature > FINAL_TEMPERATURE:
        print(f"Temp: {temperature:.2f}, Best Cost: {best_cost:.4f}", end='\r')
        for _ in range(MAX_ITERATIONS_PER_TEMP):
            # Generate a neighbor by flipping a random bit
            neighbor_b = np.copy(current_b)
            flip_index = random.randint(0, len(neighbor_b) - 1)
            neighbor_b[flip_index] = 1 - neighbor_b[flip_index]
            
            neighbor_cost = cost_function(neighbor_b, target_goal, obstacles, robot_config, ik_config)
            
            cost_diff = neighbor_cost - current_cost
            
            # Metropolis acceptance criterion
            if cost_diff < 0 or random.random() < math.exp(-cost_diff / temperature):
                current_b = neighbor_b
                current_cost = neighbor_cost
            
            if current_cost < best_cost:
                best_b = np.copy(current_b)
                best_cost = current_cost
        
        temperature *= COOLING_RATE
        cost_history.append(best_cost)
        
    print("\nSimulated Annealing finished.")
    return best_b, best_cost, cost_history

# =============================================================================
# Visualization
# =============================================================================
def plot_robot_and_cost(b: np.ndarray, g: np.ndarray, cost_history: List[float], obstacles: List[Dict], robot_config: Dict, env_config: Dict):
    """
    Visualizes the final robot configuration, cost, and environment.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
    
    # --- Plot 1: Final Robot Configuration ---
    all_points = fk(b, robot_config, return_all_coords=True)
    
    # Plot obstacles from env_config
    for obs in obstacles:
        obs_type = obs.get('type')
        if obs_type == 'rectangle':
            v = np.array(obs['vertices'])
            rect = patches.Rectangle(v[0], v[1,0]-v[0,0], v[1,1]-v[0,1], linewidth=1, edgecolor='r', facecolor='salmon', alpha=0.7, label='Obstacle')
            ax1.add_patch(rect)
        elif obs_type == 'circle':
            center = obs['center']
            radius = obs['radius']
            circle = patches.Circle(center, radius, linewidth=1, edgecolor='r', facecolor='salmon', alpha=0.7, label='Obstacle')
            ax1.add_patch(circle)

    # Plot robot body
    ax1.plot(all_points[:, 0], all_points[:, 1], 'o-', label='Robot Body', color='royalblue', linewidth=2, markersize=4)
    
    # Plot key points
    ax1.plot(0, 0, 'ks', markersize=10, label='Base')
    ax1.plot(g[0], g[1], 'g*', markersize=15, label='Target')
    ax1.plot(all_points[-1, 0], all_points[-1, 1], 'ro', markersize=10, label='End-Effector')
    
    # Style the plot
    ax1.set_title(f"Final Robot Configuration (N={robot_config['joints_number']})", fontsize=14)
    ax1.set_xlabel('X Position (mm)', fontsize=12)
    ax1.set_ylabel('Z Position (mm)', fontsize=12) # Changed label to Z
    
    # Set plot bounds from env_config
    bounds = env_config.get('bounds', {'x': [-500, 500], 'z': [0, 1000]})
    ax1.set_xlim(bounds['x'])
    ax1.set_ylim(bounds['z'])
    
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), fontsize=10)
    
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_aspect('equal', adjustable='box')

    # --- Plot 2: Cost Convergence ---
    ax2.plot(cost_history, color='orangered', linewidth=2)
    ax2.set_title('Cost Convergence Over Time', fontsize=14)
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
        with open('robot_config.yaml', 'r') as file:
            robot_config = yaml.safe_load(file)
        with open('env.yaml', 'r') as file:
            env_config = yaml.safe_load(file)
        with open('ik_config.yaml', 'r') as file:
            ik_config = yaml.safe_load(file)
    except FileNotFoundError as e:
        print(f"Error: Configuration file not found. Make sure robot_config.yaml, env.yaml, and ik_config.yaml are in the same directory. {e}")
        exit()
    
    GOAL = np.array(env_config.get('goal', [0, 800]))
    OBSTACLES = env_config.get('obstacles', [])

    print(f"Attempting to reach target (X, Z): {GOAL} while avoiding obstacles.")

    optimal_b, final_cost, history = simulated_annealing(GOAL, OBSTACLES, robot_config, ik_config)
    
    final_points = fk(optimal_b, robot_config, return_all_coords=True)
    final_position = final_points[-1]
    final_error = np.linalg.norm(final_position - GOAL)
    
    print("\n" + "="*30)
    print("          Final Results")
    print("="*30)
    print(f"Target Position (X, Z):    {GOAL}")
    print(f"Final Position (X, Z):     {np.round(final_position, 4)}")
    print(f"Final Cost:                {final_cost:.6f}")
    print(f"Final Distance Err:        {final_error:.6f}")
    print(f"Collision in final state:  {check_collision(final_points, OBSTACLES)}")
    print("\nOptimal Actuator State (b):")
    print(optimal_b.reshape(robot_config['joints_number'], 2))
    
    plot_robot_and_cost(optimal_b, GOAL, history, OBSTACLES, robot_config, env_config)
