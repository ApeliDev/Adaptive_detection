import carla
import gymnasium as gym
import numpy as np
import cv2
import torch
import time
import random
from typing import Dict, Tuple, Optional
from gymnasium import spaces
from collections import deque

from .multi_detector import MultiDetector
from .vit_encoder import ViTEncoder
from .adaptive_roi import AdaptiveROI

class CarlaMARLEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, config_path: str = "configs/defaults.yaml"):
        super().__init__()
        
        # Load config
        self.config = self._load_config(config_path)
        
        # Connect to CARLA
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        
        # multi-agent observation space 
        self.observation_space = spaces.Dict({
            "visual": spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8),
            "vehicle": spaces.Dict({
                "speed": spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
                "acceleration": spaces.Box(low=-10, high=10, shape=(3,), dtype=np.float32),
                "steering": spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                "throttle": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "brake": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            }),
            "perception": spaces.Dict({
                "object_density": spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
                "avg_confidence": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "fps": spaces.Box(low=0, high=60, shape=(1,), dtype=np.float32),
            }),
            "computational": spaces.Dict({
                "inference_time": spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
            })
        })
        
        # Multi-agent action space
        self.action_space = spaces.Dict({
            "resolution": spaces.Box(low=0.5, high=1.0, shape=(1,), dtype=np.float32),
            "roi": spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32),  # x1,y1,x2,y2
        })
        
        # Initialize components
        self.detector = MultiDetector(self.config['detector'])
        self.vit_encoder = ViTEncoder()
        self.roi_selector = AdaptiveROI()
        self.vehicle = None
        self.camera = None
        self.collision_sensor = None
        
        # State tracking
        self.current_image = None
        self.last_detections = []
        self.frame_times = deque(maxlen=30)
        self.episode_step = 0
        self.reset()
    
    def reset(self, seed=None, options=None):
        # Clean up previous episode
        self._cleanup()
        
        # Spawn vehicle
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('model3')[0]
        spawn_points = self.world.get_map().get_spawn_points()
        self.vehicle = self.world.spawn_actor(vehicle_bp, random.choice(spawn_points))
        
        # Setup camera
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '640')
        camera_bp.set_attribute('image_size_y', '480')
        self.camera = self.world.spawn_actor(
            camera_bp,
            carla.Transform(carla.Location(x=1.5, z=2.4)),
            attach_to=self.vehicle
        )
        self.camera.listen(self._process_image)
        
        # Setup collision sensor
        collision_bp = blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(
            collision_bp,
            carla.Transform(),
            attach_to=self.vehicle
        )
        self.collision_occurred = False
        self.collision_sensor.listen(self._on_collision)
        
        # Reset state
        self.episode_step = 0
        self.frame_times.clear()
        self.last_detections = []
        
        return self._get_observation(), {}
    
    def step(self, action):
        start_time = time.time()
        self.episode_step += 1
        
        # Apply actions
        roi_image = self._apply_roi_action(action['roi'], action['resolution'])
        
        # Perform detection
        detection_results = self.detector.detect(roi_image)
        self.last_detections = detection_results['detections']
        
        # Calculate reward
        reward = self._calculate_reward(detection_results)
        
        # Update frame timing
        self.frame_times.append(time.time() - start_time)
        
        # Get observation
        obs = self._get_observation()
        
        # Check termination
        terminated = self.collision_occurred or self.episode_step >= self.config['environment']['max_episode_steps']
        truncated = False
        
        return obs, reward, terminated, truncated, {}
    
    def _get_observation(self):
        """Construct multi-agent observation"""
        # Vehicle state
        velocity = self.vehicle.get_velocity()
        speed = np.linalg.norm([velocity.x, velocity.y, velocity.z]) * 3.6  # km/h
        acceleration = self.vehicle.get_acceleration()
        control = self.vehicle.get_control()
        
        # Perception state
        avg_conf = np.mean([d['confidence'] for d in self.last_detections]) if self.last_detections else 0.0
        fps = 1.0 / np.mean(self.frame_times) if self.frame_times else 30.0
        
        # Visual observation
        visual_obs = cv2.resize(self.current_image, (224, 224)) if self.current_image is not None else np.zeros((224,224,3))
        
        return {
            "visual": visual_obs,
            "vehicle": {
                "speed": np.array([speed], dtype=np.float32),
                "acceleration": np.array([acceleration.x, acceleration.y, acceleration.z], dtype=np.float32),
                "steering": np.array([control.steer], dtype=np.float32),
                "throttle": np.array([control.throttle], dtype=np.float32),
                "brake": np.array([control.brake], dtype=np.float32),
            },
            "perception": {
                "object_density": np.array([len(self.last_detections)], dtype=np.float32),
                "avg_confidence": np.array([avg_conf], dtype=np.float32),
                "fps": np.array([fps], dtype=np.float32),
            },
            "computational": {
                "inference_time": np.array([self.frame_times[-1] * 1000 if self.frame_times else 0], dtype=np.float32),
            }
        }
    
    def _calculate_reward(self, detection_results):
        """Calculate multi-component reward"""
        reward = 0.0
        cfg = self.config['environment']['reward_config']
        
        # Detection reward
        reward += len(detection_results['detections']) * cfg['detection_reward']
        
        # FPS reward
        fps = 1.0 / detection_results['metrics']['inference_time'] * 1000
        if fps > 20:
            reward += cfg['fps_reward']
        elif fps < 10:
            reward -= cfg['fps_reward']
        
        # Collision penalty
        if self.collision_occurred:
            reward += cfg['collision_penalty']
        
        return reward
    
    def _apply_roi_action(self, roi_coords, scale):
        """Apply ROI selection and scaling"""
        if self.current_image is None:
            return np.zeros((224,224,3), dtype=np.uint8)
        
        # Apply ROI
        h, w = self.current_image.shape[:2]
        x1, y1, x2, y2 = roi_coords
        roi_img = self.current_image[
            int(y1*h):int(y2*h),
            int(x1*w):int(x2*w)
        ]
        
        # Apply scaling
        scale = np.clip(scale, 0.5, 1.0)
        return cv2.resize(roi_img, (0,0), fx=scale, fy=scale)
    
    def _process_image(self, image):
        """Callback for camera sensor"""
        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))
        self.current_image = array[:, :, :3]  
    
    def _on_collision(self, event):
        """Callback for collision sensor"""
        self.collision_occurred = True
    
    def _cleanup(self):
        """Clean up CARLA actors"""
        if self.vehicle is not None:
            self.vehicle.destroy()
        if self.camera is not None:
            self.camera.destroy()
        if self.collision_sensor is not None:
            self.collision_sensor.destroy()
    
    def _load_config(self, path):
        """Load configuration file"""
        import yaml
        with open(path, 'r') as f:
            return yaml.safe_load(f)