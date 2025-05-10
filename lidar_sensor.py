import numpy as np
import mujoco
import matplotlib.pyplot as plt


class LidarSensor:
    """Simulated LiDAR sensor using MuJoCo's raycasting functionality"""

    def __init__(self, model, data, lidar_name="velodyne",
                 num_rays_h=360, num_rays_v=16,
                 h_fov=2*np.pi, v_fov=np.pi/6,
                 max_range=10.0):
        """
        Initialize the LiDAR sensor.

        Args:
            model: MuJoCo model
            data: MuJoCo data
            lidar_name: name of the LiDAR body in the model
            num_rays_h: number of horizontal rays
            num_rays_v: number of vertical rays (elevation layers)
            h_fov: horizontal field of view in radians
            v_fov: vertical field of view in radians
            max_range: maximum detection range
        """
        self.model = model
        self.data = data
        self.num_rays_h = num_rays_h
        self.num_rays_v = num_rays_v
        self.max_range = max_range

        # Find the LiDAR body ID
        self.lidar_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, lidar_name)
        if self.lidar_id < 0:
            raise ValueError(
                f"Body with name '{lidar_name}' not found in model")

        # Calculate ray angles
        self.h_angles = np.linspace(0, h_fov, num_rays_h, endpoint=False)

        # For VLP-16 style, vertical angles are fixed at specific elevations
        if num_rays_v == 16:  # VLP-16 style
            # VLP-16 has specific vertical angles (in degrees)
            vlp16_v_angles = np.array([
                -15, -13, -11, -9, -7, -5, -3, -1,
                1, 3, 5, 7, 9, 11, 13, 15
            ])
            self.v_angles = np.radians(vlp16_v_angles)
        else:
            # Generic vertical FOV distribution
            v_min = -v_fov/2
            v_max = v_fov/2
            self.v_angles = np.linspace(v_min, v_max, num_rays_v)

        # Pre-allocate results array (2D: vertical x horizontal)
        self.ranges = np.zeros((num_rays_v, num_rays_h))
        self.points = np.zeros((num_rays_v, num_rays_h, 3))
        self.intensities = np.zeros((num_rays_v, num_rays_h))

        # For visualization
        self.fig = None
        self.ax = None

    def update(self):
        """Cast rays and update range measurements"""
        # Get the LiDAR position and orientation
        lidar_pos = self.data.xpos[self.lidar_id].copy()
        lidar_mat = self.data.xmat[self.lidar_id].reshape(3, 3).copy()

        # For all vertical angles
        for v_idx, v_angle in enumerate(self.v_angles):
            # For all horizontal angles
            for h_idx, h_angle in enumerate(self.h_angles):
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

                # FIXED: Call mj_ray with the correct signature
                # This returns the distance directly
                ray_dist = mujoco.mj_ray(
                    self.model,
                    self.data,
                    lidar_pos,          # Start position
                    ray_dir_world,      # Direction
                    None,               # No specific geom group to collide with
                    1,                  # Collision with all geoms
                    -1,                 # Ignore the body of the robot itself
                    ray_geom,           # Output: geom that was hit
                )

                # Store the range and compute hit point
                if ray_geom[0] >= 0 and ray_dist < self.max_range:  # If ray hit something
                    self.ranges[v_idx, h_idx] = ray_dist
                    # Compute hit position
                    self.points[v_idx, h_idx] = lidar_pos + \
                        ray_dist * ray_dir_world

                    # Calculate simple intensity based on distance (closer = higher intensity)
                    # Real LiDAR intensity depends on material properties and incidence angle
                    self.intensities[v_idx, h_idx] = max(
                        0, 1.0 - ray_dist/self.max_range)
                else:
                    # No hit or hit beyond max range, set to max range and point at max range
                    self.ranges[v_idx, h_idx] = self.max_range
                    self.points[v_idx, h_idx] = lidar_pos + \
                        self.max_range * ray_dir_world
                    self.intensities[v_idx, h_idx] = 0.0

        return self.ranges.copy()

    def get_point_cloud(self):
        """Return the point cloud as a flattened array of points"""
        # Filter out max-range points
        mask = self.ranges < self.max_range
        valid_points = self.points[mask]
        return valid_points

    def visualize_scan(self, ax=None):
        """Visualize the LiDAR scan as a top-down view (middle layer)"""
        if ax is None:
            if self.fig is None:
                self.fig, self.ax = plt.subplots(
                    figsize=(8, 8), subplot_kw={'projection': 'polar'})
            ax = self.ax

        # Clear previous plot
        ax.clear()

        # Use middle vertical layer for 2D visualization
        middle_idx = len(self.v_angles) // 2

        # Plot the scan
        ax.scatter(self.h_angles, self.ranges[middle_idx],
                   c=self.intensities[middle_idx], cmap='jet', s=5)
        ax.set_rmax(self.max_range)
        ax.set_title('LiDAR Scan (Top View)')

        plt.draw()
        plt.pause(0.001)

        return ax
