# vlp16_sensor.py
import numpy as np
import mujoco
import time
import asyncio
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import threading
from concurrent.futures import ThreadPoolExecutor


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

        # Variables for async operation
        self.running = False
        self.update_task = None
        self.lock = threading.RLock()  # Reentrant lock for thread safety
        # For running blocking operations
        self.executor = ThreadPoolExecutor(max_workers=1)

        # Visualization variables
        self.fig = None
        self.ax = None
        self.scatter = None
        self.animation = None
        self._current_points = None
        self._current_lidar_pos = None

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
        with self.lock:
            # Get the LiDAR position and orientation
            lidar_pos = self.data.xpos[self.lidar_id].copy()
            lidar_mat = self.data.xmat[self.lidar_id].reshape(3, 3).copy()

            # Define minimum detection distance (blind spot)
            min_detection_dist = 0.1  # 10cm blind spot

            # Offset distance from LiDAR center to start ray casting
            offset_dist = 0.3  # 30cm offset to ensure we're outside the robot

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
                    ray_dir_world = ray_dir_world / \
                        np.linalg.norm(ray_dir_world)

                    # Offset ray start position outside the robot
                    ray_start_pos = lidar_pos + offset_dist * ray_dir_world

                    # Initialize output arrays for raycasting
                    ray_geom = np.zeros(1, dtype=np.int32)

                    # Cast the ray
                    ray_dist = mujoco.mj_ray(
                        self.model,
                        self.data,
                        # Start position (offset from LiDAR center)
                        ray_start_pos,
                        ray_dir_world,      # Direction
                        None,               # No specific geom group to collide with
                        1,                  # Collision with all geoms
                        self.lidar_id,      # Exclude LiDAR body itself
                        ray_geom            # Output: geom that was hit
                    )

                    # If we hit something within range
                    if ray_geom[0] >= 0 and ray_dist < self.max_range:
                        # Get the body ID of the hit geometry to check if it's part of the robot
                        hit_body = self.model.geom_bodyid[ray_geom[0]]

                        # Check if it's a robot part by looking at parent hierarchy
                        is_robot_part = False
                        parent_id = hit_body

                        # Check parent hierarchy
                        while parent_id > 0:  # 0 is the world body
                            body_name = mujoco.mj_id2name(
                                self.model, mujoco.mjtObj.mjOBJ_BODY, parent_id)
                            if body_name and ("base_link" in body_name or "chassis" in body_name):
                                is_robot_part = True
                                break
                            parent_id = self.model.body_parentid[parent_id]

                        if not is_robot_part:
                            # Calculate total distance from original LiDAR center
                            total_dist = ray_dist + offset_dist

                            # Skip hits that are too close (within blind spot)
                            if total_dist <= min_detection_dist:
                                self.ranges[v_idx, h_idx] = self.max_range
                                self.points[v_idx, h_idx] = lidar_pos + \
                                    self.max_range * ray_dir_world
                                self.intensities[v_idx, h_idx] = 0.0
                            else:
                                # Add noise to simulate sensor imperfections
                                # 3cm standard deviation
                                noise = np.random.normal(0, 0.03)
                                measured_dist = max(0, total_dist + noise)

                                self.ranges[v_idx, h_idx] = measured_dist
                                self.points[v_idx, h_idx] = lidar_pos + \
                                    measured_dist * ray_dir_world
                                self.intensities[v_idx, h_idx] = max(
                                    0, 1.0 - measured_dist/self.max_range)
                        else:
                            # Hit was on robot part, treat as no hit
                            self.ranges[v_idx, h_idx] = self.max_range
                            self.points[v_idx, h_idx] = lidar_pos + \
                                self.max_range * ray_dir_world
                            self.intensities[v_idx, h_idx] = 0.0
                    else:
                        # No hit or hit beyond max range
                        self.ranges[v_idx, h_idx] = self.max_range
                        self.points[v_idx, h_idx] = lidar_pos + \
                            self.max_range * ray_dir_world
                        self.intensities[v_idx, h_idx] = 0.0

            # Update the current points for visualization
            points = self.get_point_cloud(filter_max_range=True)
            self._current_points = points.copy() if len(points) > 0 else None
            self._current_lidar_pos = lidar_pos.copy()

            self.last_update_time = time.time()

            return {
                'ranges': self.ranges.copy(),
                'points': points,
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
        with self.lock:
            if filter_max_range:
                # Filter out max-range points
                mask = self.ranges < self.max_range
                valid_points = self.points[mask]
                return valid_points
            else:
                # Return all points
                return self.points.reshape(-1, 3)

    def get_statistics(self):
        """Return statistics about the LiDAR scan"""
        with self.lock:
            total_rays = self.num_rays_h * self.num_rays_v
            valid_points = np.sum(self.ranges < self.max_range)
            hit_percentage = (valid_points / total_rays) * 100

            # Calculate distance statistics (for valid points only)
            valid_ranges = self.ranges[self.ranges < self.max_range]
            min_distance = np.min(
                valid_ranges) if valid_points > 0 else self.max_range
            max_distance = np.max(valid_ranges) if valid_points > 0 else 0
            mean_distance = np.mean(valid_ranges) if valid_points > 0 else 0

            return {
                'total_rays': total_rays,
                'valid_points': valid_points,
                'hit_percentage': hit_percentage,
                'min_distance': min_distance,
                'max_distance': max_distance,
                'mean_distance': mean_distance
            }

    def save_point_cloud_ply(self, filename):
        """Save the point cloud to a PLY file"""
        # Get the point cloud (filtered to remove max range points)
        points = self.get_point_cloud(filter_max_range=True)

        if len(points) == 0:
            print("No points to save (all points are at max range)")
            return

        # Create PLY header
        header = [
            "ply",
            "format ascii 1.0",
            f"element vertex {len(points)}",
            "property float x",
            "property float y",
            "property float z",
            "end_header"
        ]

        # Write the PLY file
        with open(filename, 'w') as f:
            f.write('\n'.join(header) + '\n')
            for point in points:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")

        print(f"Saved {len(points)} points to {filename}")

    def visualize_point_cloud(self, ax=None, color='viridis', point_size=1, show_lidar=True,
                              equal_aspect=True, elev=30, azim=45):
        """
        Visualize the point cloud in 3D using matplotlib.

        Args:
            ax: A matplotlib 3D axis. If None, a new figure and axis will be created
            color: Color map for the points. Default is 'viridis'
            point_size: Size of the plotted points. Default is 1
            show_lidar: If True, shows the LiDAR position as a red marker
            equal_aspect: If True, sets equal aspect ratio for all axes
            elev: Elevation viewing angle (degrees)
            azim: Azimuth viewing angle (degrees)

        Returns:
            ax: The matplotlib 3D axis with the point cloud plotted
        """
        # Get the point cloud (filtered to remove points at max range)
        points = self.get_point_cloud(filter_max_range=True)

        # Create a new figure and axis if none was provided
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

        # Check if we have any points to plot
        if len(points) == 0:
            print("No points to plot (all points are at max range or no data yet)")
            return ax

        # Get the positions of the points
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        # Color the points based on their distance from the LiDAR
        lidar_pos = self.data.xpos[self.lidar_id].copy()
        distances = np.sqrt(np.sum((points - lidar_pos) ** 2, axis=1))

        # Plot the points
        scatter = ax.scatter(x, y, z, c=distances,
                             cmap=color, s=point_size, alpha=0.8)

        # Mark the LiDAR position
        if show_lidar:
            ax.scatter([lidar_pos[0]], [lidar_pos[1]], [lidar_pos[2]],
                       c='red', s=50, marker='*', label='LiDAR')
            ax.legend()

        # Add a colorbar
        plt.colorbar(scatter, ax=ax, label='Distance (m)')

        # Set labels and title
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('VLP-16 LiDAR Point Cloud')

        # Set view angle
        ax.view_init(elev=elev, azim=azim)

        # Set equal aspect ratio if requested
        if equal_aspect:
            # Get the limits of the data
            max_range = max([
                np.max(x) - np.min(x) if len(x) > 0 else 1,
                np.max(y) - np.min(y) if len(y) > 0 else 1,
                np.max(z) - np.min(z) if len(z) > 0 else 1
            ])

            if len(x) > 0:
                mid_x = (np.max(x) + np.min(x)) * 0.5
                mid_y = (np.max(y) + np.min(y)) * 0.5
                mid_z = (np.max(z) + np.min(z)) * 0.5

                ax.set_xlim(mid_x - max_range * 0.5, mid_x + max_range * 0.5)
                ax.set_ylim(mid_y - max_range * 0.5, mid_y + max_range * 0.5)
                ax.set_zlim(mid_z - max_range * 0.5, mid_z + max_range * 0.5)

        plt.tight_layout()

        return ax

    async def start_async(self, visualize=True, update_rate=None, visualization_rate=10):
        """
        Start the LiDAR sensor in asynchronous mode.

        Args:
            visualize: If True, starts the visualization window
            update_rate: The update rate in Hz. If None, uses rotation_rate
            visualization_rate: The visualization update rate in Hz
        """
        if self.running:
            print("LiDAR sensor is already running")
            return

        self.running = True

        # Use rotation rate as update rate if not specified
        if update_rate is None:
            update_rate = self.rotation_rate

        # Start update task for LiDAR ray casting
        self.update_task = asyncio.create_task(self._update_task(update_rate))

        # Create visualization if requested - this runs in the main thread
        if visualize:
            self._setup_visualization(visualization_rate)

        print(
            f"Started LiDAR sensor in async mode (update: {update_rate} Hz, viz: {visualization_rate if visualize else 0} Hz)")

    async def stop_async(self):
        """Stop the LiDAR sensor's asynchronous tasks"""
        if not self.running:
            return

        self.running = False

        # Cancel update task if it exists
        if self.update_task:
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
            self.update_task = None

        # Close figure if it exists
        if self.fig:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            self.scatter = None
            self.animation = None

        print("Stopped LiDAR sensor async tasks")

    async def _update_task(self, rate_hz):
        """Background task that updates the LiDAR data at the specified rate"""
        try:
            period = 1.0 / rate_hz
            while self.running:
                start_time = time.time()

                # Run update in the thread pool to avoid blocking the event loop
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(self.executor, self.update)

                # Sleep for the remainder of the period
                elapsed = time.time() - start_time
                sleep_time = max(0, period - elapsed)
                await asyncio.sleep(sleep_time)
        except asyncio.CancelledError:
            # Task was cancelled, clean up
            pass
        except Exception as e:
            print(f"Error in LiDAR update task: {e}")
            self.running = False

    def _setup_visualization(self, frame_rate):
        """Setup the LiDAR visualization using matplotlib's animation framework"""
        # Create the figure and axis - this runs in the main thread
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Initial empty plot
        self.scatter = self.ax.scatter(
            [], [], [], c=[], cmap='viridis', s=1, alpha=0.8)
        self.lidar_marker = self.ax.scatter(
            [], [], [], c='red', s=50, marker='*', label='LiDAR')

        # Set labels
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title('VLP-16 LiDAR Point Cloud')

        # Use FuncAnimation for proper animation in the main thread
        interval = int(1000 / frame_rate)  # convert frame rate to milliseconds
        self.animation = FuncAnimation(
            self.fig, self._update_animation_frame,
            interval=interval, blit=False, save_count=0)

        # Show the plot (non-blocking)
        plt.ion()  # Turn on interactive mode
        plt.show(block=False)

    def _update_animation_frame(self, frame):
        """Update function for the animation - runs in the main thread"""
        # Clear the previous plot
        self.ax.clear()

        # Get the current points (safely)
        with self.lock:
            points = self._current_points
            lidar_pos = self._current_lidar_pos

        if points is None or len(points) == 0 or lidar_pos is None:
            self.ax.set_title('No points detected')
            return

        # Get the positions of the points
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        # Color the points based on their distance from the LiDAR
        distances = np.sqrt(np.sum((points - lidar_pos) ** 2, axis=1))

        # Plot the points
        self.scatter = self.ax.scatter(
            x, y, z, c=distances, cmap='viridis', s=1, alpha=0.8)

        # Mark the LiDAR position
        self.ax.scatter([lidar_pos[0]], [lidar_pos[1]], [lidar_pos[2]],
                        c='red', s=50, marker='*', label='LiDAR')

        # Set labels and title
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title(f'VLP-16 LiDAR Point Cloud - {len(points)} points')

        # Get the limits of the data for equal aspect
        max_range = max([
            np.max(x) - np.min(x) if len(x) > 0 else 1,
            np.max(y) - np.min(y) if len(y) > 0 else 1,
            np.max(z) - np.min(z) if len(z) > 0 else 1
        ])

        # Set view limits
        if len(x) > 0:
            mid_x = (np.max(x) + np.min(x)) * 0.5
            mid_y = (np.max(y) + np.min(y)) * 0.5
            mid_z = (np.max(z) + np.min(z)) * 0.5

            self.ax.set_xlim(mid_x - max_range * 0.5, mid_x + max_range * 0.5)
            self.ax.set_ylim(mid_y - max_range * 0.5, mid_y + max_range * 0.5)
            self.ax.set_zlim(mid_z - max_range * 0.5, mid_z + max_range * 0.5)

        # Let matplotlib render the frame
        self.fig.canvas.draw_idle()
