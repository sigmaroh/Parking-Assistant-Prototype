"""Car simulation module with MPC controller."""

import numpy as np
from typing import List, Tuple
from config import SimulationConfig
from mpc_controller import Environment, Linear_MPC_Controller, Car_Dynamics


class CarSimulator:
    """Simulates car movement along a path using MPC control."""
    
    def __init__(self):
        self.car_dynamics = None
        self.mpc_controller = None
        self.mpc_env = None
        
        # Car state
        self.x = 0.0
        self.y = 0.0
        self.velocity = SimulationConfig.DEFAULT_VELOCITY
        self.psi = 0.0  # Heading in radians
        self.delta = 0.0  # Steering angle
        
        # Configuration
        self.wheelbase = SimulationConfig.CAR_WHEELBASE
        self.max_steering = SimulationConfig.MAX_STEERING_ANGLE
        self.horizon = SimulationConfig.MPC_HORIZON
    
    def initialize(self, start_pos: Tuple[float, float], 
                   path: List[Tuple[float, float]],
                   obstacles: np.ndarray = None):
        """
        Initialize simulator for a new simulation.
        
        Args:
            start_pos: Starting position (row, col)
            path: Reference path to follow
            obstacles: Obstacle positions for MPC environment
        """
        # Set initial position (convert from grid to continuous)
        self.x = float(start_pos[1])  # col -> x
        self.y = float(start_pos[0])  # row -> y
        
        # Calculate initial heading towards next waypoint
        if len(path) > 1:
            next_pos = path[1]
            dx = next_pos[1] - start_pos[1]
            dy = next_pos[0] - start_pos[0]
            self.psi = np.arctan2(dy, dx)
        else:
            self.psi = 0.0
        
        self.delta = 0.0
        self.velocity = SimulationConfig.DEFAULT_VELOCITY
        
        # Initialize car dynamics
        self.car_dynamics = Car_Dynamics(
            x_0=self.x,
            y_0=self.y,
            v_0=self.velocity,
            psi_0=self.psi,
            length=self.wheelbase,
            dt=SimulationConfig.SIMULATION_DT
        )
        
        # Initialize MPC controller
        self.mpc_controller = Linear_MPC_Controller()
        
        # Initialize environment
        if obstacles is None:
            obstacles = np.array([[0, 0]])
        self.mpc_env = Environment(obstacles)
        
        # Draw reference path
        path_for_mpc = [(p[1], p[0]) for p in path]  # Convert to (x, y)
        self.mpc_env.draw_path(path_for_mpc)
    
    def step(self, reference_points: np.ndarray) -> bool:
        """
        Perform one simulation step using MPC control.
        
        Args:
            reference_points: Array of reference points for MPC horizon
            
        Returns:
            True if step was successful, False otherwise
        """
        try:
            # Get optimal control from MPC
            acceleration, self.delta = self.mpc_controller.optimize(
                self.car_dynamics, reference_points
            )
            
            # Update car dynamics
            state_dot = self.car_dynamics.move(acceleration, self.delta)
            self.car_dynamics.update_state(state_dot)
            
            # Update local state variables
            self.x = self.car_dynamics.x
            self.y = self.car_dynamics.y
            self.velocity = self.car_dynamics.v
            self.psi = self.car_dynamics.psi
            
            return True
            
        except Exception as e:
            print(f"MPC step failed: {e}")
            return False
    
    def step_pure_pursuit(self, target_point: Tuple[float, float], dt: float):
        """
        Fallback control using pure pursuit algorithm.
        
        Args:
            target_point: Target point to track (row, col)
            dt: Time step
        """
        target_x = target_point[1]  # col -> x
        target_y = target_point[0]  # row -> y
        
        # Transform to vehicle coordinates
        dx = target_x - self.x
        dy = target_y - self.y
        
        local_x = dx * np.cos(-self.psi) - dy * np.sin(-self.psi)
        local_y = dx * np.sin(-self.psi) + dy * np.cos(-self.psi)
        
        # Calculate curvature
        L_d = np.sqrt(local_x**2 + local_y**2)
        if L_d < 0.01:
            self.delta = 0.0
        else:
            curvature = 2 * local_y / (L_d ** 2)
            self.delta = np.arctan(self.wheelbase * curvature)
            self.delta = np.clip(self.delta, -self.max_steering, self.max_steering)
        
        # Update state using bicycle model
        self.x += self.velocity * np.cos(self.psi) * dt
        self.y += self.velocity * np.sin(self.psi) * dt
        self.psi += (self.velocity / self.wheelbase) * np.tan(self.delta) * dt
        
        # Normalize heading
        self.psi = np.arctan2(np.sin(self.psi), np.cos(self.psi))
    
    def get_state(self) -> dict:
        """Get current car state."""
        return {
            'x': self.x,
            'y': self.y,
            'velocity': self.velocity,
            'psi': self.psi,
            'delta': self.delta,
            'heading_deg': np.degrees(self.psi)
        }
    
    def distance_to_goal(self, goal: Tuple[float, float]) -> float:
        """Calculate distance to goal position."""
        goal_x = goal[1]  # col -> x
        goal_y = goal[0]  # row -> y
        return np.sqrt((self.x - goal_x)**2 + (self.y - goal_y)**2)
    
    def find_closest_path_index(self, path: List[Tuple[float, float]]) -> int:
        """Find index of closest point on path."""
        min_dist = float('inf')
        closest_idx = 0
        
        for i, p in enumerate(path):
            d = np.sqrt((self.x - p[1])**2 + (self.y - p[0])**2)
            if d < min_dist:
                min_dist = d
                closest_idx = i
        
        return closest_idx
    
    def get_reference_points(self, path: List[Tuple[float, float]], 
                            start_idx: int) -> np.ndarray:
        """
        Get reference points for MPC horizon.
        
        Args:
            path: Reference path
            start_idx: Starting index in path
            
        Returns:
            Array of reference points (horizon x 2)
        """
        reference_points = []
        for i in range(self.horizon):
            idx = min(start_idx + i, len(path) - 1)
            ref_point = path[idx]
            reference_points.append([ref_point[1], ref_point[0]])  # (x, y)
        
        return np.array(reference_points)


class CarRenderer:
    """Renders car visualization."""
    
    @staticmethod
    def get_car_dimensions(parking_spots: List[dict], 
                          cell_width: int, cell_height: int) -> Tuple[int, int]:
        """
        Calculate car dimensions based on parking spots.
        
        Returns:
            (car_width, car_length) in pixels
        """
        if parking_spots and len(parking_spots) > 0:
            avg_spot_width = sum(
                s['bounding_rect']['width'] for s in parking_spots
            ) / len(parking_spots)
            avg_spot_height = sum(
                s['bounding_rect']['height'] for s in parking_spots
            ) / len(parking_spots)
            
            spot_min = min(avg_spot_width, avg_spot_height)
            spot_max = max(avg_spot_width, avg_spot_height)
            
            car_width = int(spot_min * SimulationConfig.CAR_WIDTH_RATIO)
            car_length = int(spot_max * SimulationConfig.CAR_LENGTH_RATIO)
        else:
            car_length = max(SimulationConfig.FALLBACK_CAR_LENGTH, cell_height * 2)
            car_width = max(SimulationConfig.FALLBACK_CAR_WIDTH, cell_width)
        
        return car_width, car_length
    
    @staticmethod
    def rotate_points(points: List[Tuple[float, float]], 
                     angle: float, 
                     center: Tuple[float, float]) -> List[Tuple[int, int]]:
        """Rotate points around a center by an angle."""
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        rotated = []
        
        for px, py in points:
            rx = center[0] + (px - center[0]) * cos_a - (py - center[1]) * sin_a
            ry = center[1] + (px - center[0]) * sin_a + (py - center[1]) * cos_a
            rotated.append((int(rx), int(ry)))
        
        return rotated

