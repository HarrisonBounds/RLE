# vlp16_sensor.py
import numpy as np
import mujoco
import time


class VLP16Sensor:
    """Simulated Velodyne Puck (VLP-16) LiDAR sensor using MuJoCo's raycasting"""

    def __init__(self, model, data, lidar_name="velodyne",
                 horizontal_resolution=1.0,  # degrees (0.1° to 0.4° per specs)
                 rotation_rate=10,           # Hz (5-20 Hz per specs)
                 max_range=100.0):           # meters
        """
        Initialize the VLP-16 LiDAR sensor.

        Args:
            model: MuJoCo model
            data: MuJoCo data
            lidar_name: name of the LiDAR body in the model
            horizontal_resolution: angular resolution in degrees (0.1° to 0.4° per specs)
            rotation_rate: rotation rate in Hz (5-20 Hz per specs)
            max_range: maximum detection range in meters (100m per specs)
        """
        self.model = model
        self.data = data
        self.max_range = max_range
        self.rotation_rate = rotation_rate
        self.last_update_time = time.time()

        # Horizontal scan parameters
        self.horizontal_resolution_deg = horizontal_resolution
        self.horizontal_resolution_rad = np.radians(horizontal_resolution)
        self.num_rays_h = int(360 / horizontal_resolution)

        # Calculate complete horizontal ray angles (full 360°)
        self.h_angles = np.linspace(
            0, 2*np.pi, self.num_rays_h, endpoint=False)

        # VLP-16 has exactly 16 vertical channels with specific fixed angles
        self.num_rays_v = 16

        # VLP-16's 16 channels are at these exact elevation angles (in degrees)
        vlp16_v_angles = np.array([
            -15, -13, -11, -9, -7, -5, -3, -1,
            1, 3, 5, 7, 9, 11, 13, 15
        ])
        self.v_angles = np.radians(vlp16_v_angles)

        # Find the LiDAR body ID
        self.lidar_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, lidar_name)

        if self.lidar_id < 0:
            raise ValueError(
                f"Body with name '{lidar_name}' not found in model")

        # Pre-allocate results arrays (2D: vertical x horizontal)
        self.ranges = np.ones(
            (self.num_rays_v, self.num_rays_h)) * self.max_range
        self.points = np.zeros((self.num_rays_v, self.num_rays_h, 3))
        self.intensities = np.zeros((self.num_rays_v, self.num_rays_h))

        print(
            f"VLP-16 LiDAR initialized with {self.num_rays_v} vertical channels and {self.num_rays_h} horizontal rays")
        print(
            f"Horizontal resolution: {self.horizontal_resolution_deg}°, Rotation rate: {self.rotation_rate} Hz")

    def update(self):
        """
        Cast rays and update range measurements.

        Returns:
            Dictionary with ranges, points, and intensities
        """
        # Get the LiDAR position and orientation
        lidar_pos = self.data.xpos[self.lidar_id].copy()
        lidar_mat = self.data.xmat[self.lidar_id].reshape(3, 3).copy()

        # For all horizontal angles
        for h_idx, h_angle in enumerate(self.h_angles):
            # For all vertical angles (all 16 channels)
            for v_idx, v_angle in enumerate(self.v_angles):
                # Calculate ray direction in LiDAR frame
                # X forward, Y left, Z up in typical LiDAR frame
                ray_dir_local = np.array([
                    np.cos(v_angle) * np.cos(h_angle),  # X
                    np.cos(v_angle) * np.sin(h_angle),  # Y
                    np.sin(v_angle)                     # Z
                ])

                # Transform to world frame
                ray_dir_world = lidar_mat @ ray_dir_local

                # Normalize direction vector
                ray_dir_world = ray_dir_world / np.linalg.norm(ray_dir_world)

                # Initialize output arrays for raycasting
                ray_geom = np.zeros(1, dtype=np.int32)

                ray_dist = mujoco.mj_ray(
                    self.model,
                    self.data,
                    lidar_pos,          # Start position
                    ray_dir_world,      # Direction
                    None,               # No specific geom group to collide with
                    1,                  # Collision with all geoms
                    -1,                 # Ignore the body of the robot itself
                    ray_geom            # Output: geom that was hit
                )

                # Store the range and compute hit point
                if ray_geom[0] >= 0 and ray_dist < self.max_range:  # If ray hit something
                    # Simulate range accuracy of ±3 cm (typical)
                    noise = np.random.normal(0, 0.03)  # 3cm standard deviation
                    measured_dist = max(0, ray_dist + noise)

                    self.ranges[v_idx, h_idx] = measured_dist
                    # Compute hit position
                    self.points[v_idx, h_idx] = lidar_pos + \
                        measured_dist * ray_dir_world

                    # Calculate simplified intensity based on distance
                    self.intensities[v_idx, h_idx] = max(
                        0, 1.0 - measured_dist/self.max_range)
                else:
                    # No hit or hit beyond max range, set to max range
                    self.ranges[v_idx, h_idx] = self.max_range
                    self.points[v_idx, h_idx] = lidar_pos + \
                        self.max_range * ray_dir_world
                    self.intensities[v_idx, h_idx] = 0.0

        return {
            'ranges': self.ranges.copy(),
            'points': self.get_point_cloud(),
            'lidar_position': lidar_pos
        }

    def get_point_cloud(self, filter_max_range=True):
        """
        Return the point cloud as a flattened array of points.

        Args:
            filter_max_range: If True, filter out points at max range (no returns)

        Returns:
            Nx3 array of points, where N is the number of valid returns
        """
        if filter_max_range:
            # Filter out max-range points
            mask = self.ranges < self.max_range
            valid_points = self.points[mask]
            return valid_points
        else:
            # Return all points
            return self.points.reshape(-1, 3)
