import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List, Tuple

class DroneNavigation:
    def __init__(self, map_size: Tuple[float, float] = (10, 10), velocity: float = 10.0):
        """
        Initialize drone navigation system
        
        :param map_size: Size of the 2D map (width, height)
        :param velocity: Constant velocity of the drone
        """
        self.map_size = map_size
        self.velocity = velocity
        self.obstacles = []
        
    def add_obstacle(self, x: float, y: float, radius: float = 0.5):
        """
        Add an obstacle to the map
        
        :param x: x-coordinate of obstacle center
        :param y: y-coordinate of obstacle center
        :param radius: radius of the obstacle
        """
        self.obstacles.append((x, y, radius))
    
    class Node:
        """
        Node class for RRT algorithm
        """
        def __init__(self, x: float, y: float):
            self.x = x
            self.y = y
            self.parent = None
    
    def distance(self, node1: 'DroneNavigation.Node', node2: 'DroneNavigation.Node') -> float:
        """
        Calculate Euclidean distance between two nodes
        
        :param node1: First node
        :param node2: Second node
        :return: Distance between nodes
        """
        return np.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)
    
    def is_collision(self, node: 'DroneNavigation.Node') -> bool:
        """
        Check if a node collides with any obstacles
        
        :param node: Node to check for collision
        :return: True if collision, False otherwise
        """
        for obs_x, obs_y, radius in self.obstacles:
            if np.sqrt((node.x - obs_x)**2 + (node.y - obs_y)**2) < radius:
                return True
        return False
    
    def rrt_path_planning(self, start: Tuple[float, float], goal: Tuple[float, float], 
                           max_iterations: int = 500, step_size: float = 0.5) -> List[Node]:
        """
        RRT Path Planning Algorithm
        
        :param start: Starting coordinates (x, y)
        :param goal: Goal coordinates (x, y)
        :param max_iterations: Maximum number of iterations
        :param step_size: Step size for tree expansion
        :return: Path from start to goal
        """
        start_node = self.Node(start[0], start[1])
        goal_node = self.Node(goal[0], goal[1])
        
        tree = [start_node]
        
        for _ in range(max_iterations):
            # Random node sampling
            rand_node = self.Node(
                random.uniform(0, self.map_size[0]), 
                random.uniform(0, self.map_size[1])
            )
            
            # Find nearest node in the tree
            nearest_node = min(tree, key=lambda node: self.distance(node, rand_node))
            
            # Steer towards the random node
            angle = np.arctan2(rand_node.y - nearest_node.y, rand_node.x - nearest_node.x)
            new_node = self.Node(
                nearest_node.x + step_size * np.cos(angle),
                nearest_node.y + step_size * np.sin(angle)
            )
            
            # Check for collisions
            if not self.is_collision(new_node):
                new_node.parent = nearest_node
                tree.append(new_node)
                
                # Check if goal is reached
                if self.distance(new_node, goal_node) < step_size:
                    goal_node.parent = new_node
                    tree.append(goal_node)
                    break
        
        # Reconstruct path
        path = []
        current_node = goal_node
        while current_node:
            path.append(current_node)
            current_node = current_node.parent
        
        return list(reversed(path))
    
    def extended_kalman_filter(self, initial_state: np.ndarray, 
                                measurements: np.ndarray, 
                                control_input: np.ndarray):
        """
        Extended Kalman Filter for Drone Localization
        
        :param initial_state: Initial state [x, y, theta]
        :param measurements: Sensor measurements
        :param control_input: Control inputs (steering angle)
        :return: Estimated trajectory
        """
        # System parameters
        dt = 0.1  # Time step
        v = self.velocity  # Constant velocity
        
        # Initial state and covariance
        x = initial_state
        P = np.eye(3) * 0.1  # Initial covariance matrix
        
        # Process noise covariance
        Q = np.eye(3) * 0.01
        
        # Measurement noise covariance
        R = np.eye(2) * 0.05
        
        estimated_trajectory = [x]
        
        for u in control_input:
            # Prediction step
            # Non-linear motion model
            x_pred = x[0] + v * np.cos(x[2]) * dt
            y_pred = x[1] + v * np.sin(x[2]) * dt
            theta_pred = x[2] + u * dt
            
            x_new = np.array([x_pred, y_pred, theta_pred])
            
            # Jacobian of motion model (F matrix)
            F = np.array([
                [1, 0, -v * np.sin(x[2]) * dt],
                [0, 1, v * np.cos(x[2]) * dt],
                [0, 0, 1]
            ])
            
            # Update covariance
            P = F @ P @ F.T + Q
            
            # Update step (using available measurements)
            if len(measurements) > 0:
                # Measurement model
                z = measurements.pop(0)
                y = z - np.array([x_new[0], x_new[1]])
                
                # Measurement Jacobian (H matrix)
                H = np.array([
                    [1, 0, 0],
                    [0, 1, 0]
                ])
                
                # Kalman Gain
                S = H @ P @ H.T + R
                K = P @ H.T @ np.linalg.inv(S)
                
                # State update
                x_new = x_new + K @ y
                
                # Covariance update
                P = (np.eye(3) - K @ H) @ P
            
            x = x_new
            estimated_trajectory.append(x)
        
        return estimated_trajectory
    
    def visualize_path(self, path: List[Node]):
        """
        Visualize the RRT path and obstacles
        
        :param path: Path found by RRT
        """
        plt.figure(figsize=(10, 10))
        plt.title('Drone Navigation Path')
        plt.xlim(0, self.map_size[0])
        plt.ylim(0, self.map_size[1])
        
        # Plot obstacles
        for obs_x, obs_y, radius in self.obstacles:
            circle = plt.Circle((obs_x, obs_y), radius, color='black')
            plt.gca().add_artist(circle)
        
        # Plot path
        path_x = [node.x for node in path]
        path_y = [node.y for node in path]
        plt.plot(path_x, path_y, 'r-', linewidth=2)
        
        # Plot start and goal
        plt.plot(path_x[0], path_y[0], 'go', label='Start')
        plt.plot(path_x[-1], path_y[-1], 'ro', label='Goal')
        
        plt.legend()
        plt.grid(True)
        plt.show()

# Example usage
def main():
    # Create drone navigation system
    drone_nav = DroneNavigation()
    
    # Add obstacles
    drone_nav.add_obstacle(3, 3, 0.5)
    drone_nav.add_obstacle(5, 5, 0.5)
    drone_nav.add_obstacle(7, 2, 0.5)
    
    # Path planning
    start = (0, 0)
    goal = (9, 9)
    path = drone_nav.rrt_path_planning(start, goal)
    
    # Visualize path
    drone_nav.visualize_path(path)
    
    # Extended Kalman Filter simulation
    initial_state = np.array([0, 0, 0])  # [x, y, theta]
    control_inputs = np.random.uniform(-np.pi/12, np.pi/12, 50)  # Random steering inputs
    measurements = np.random.multivariate_normal([0, 0], np.eye(2)*0.1, 50)
    
    trajectory = drone_nav.extended_kalman_filter(initial_state, measurements, control_inputs)
    
    # Plot trajectory
    trajectory = np.array(trajectory)
    plt.figure(figsize=(10, 10))
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-')
    plt.title('Drone Trajectory Estimation')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.show()

if __name__ == "__main__":
    main()