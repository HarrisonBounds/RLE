import numpy as np
import matplotlib.pyplot as plt
from jackal_env import Jackal_Env
import time
import os

# Function to create an environment with obstacles


def create_obstacle_env():
    # Create a test environment with obstacles

    # If file exists already, no need to recreate
    if os.path.exists("jackal_obstacles.xml"):
        return "jackal_obstacles.xml"

    # Start with the base Jackal XML
    with open("jackal_velodyne.xml", "r") as f:
        xml_content = f.read()

    # Find the position to insert obstacles (before the closing worldbody tag)
    insert_pos = xml_content.find("</worldbody>")

    # Define some obstacles
    obstacles = """
        <!-- Obstacles -->
        <body name="obstacle1" pos="2 0 0.5">
            <geom type="box" size="0.5 0.5 0.5" rgba="1 0 0 1" />
        </body>
        <body name="obstacle2" pos="0 3 0.5">
            <geom type="cylinder" size="0.5 1" rgba="0 1 0 1" />
        </body>
        <body name="obstacle3" pos="-3 -2 0.5">
            <geom type="sphere" size="0.7" rgba="0 0 1 1" />
        </body>
        <body name="wall1" pos="5 0 0.5">
            <geom type="box" size="0.1 5 0.5" rgba="0.5 0.5 0.5 1" />
        </body>
        <body name="wall2" pos="-5 0 0.5">
            <geom type="box" size="0.1 5 0.5" rgba="0.5 0.5 0.5 1" />
        </body>
        <body name="wall3" pos="0 5 0.5">
            <geom type="box" size="5 0.1 0.5" rgba="0.5 0.5 0.5 1" />
        </body>
        <body name="wall4" pos="0 -5 0.5">
            <geom type="box" size="5 0.1 0.5" rgba="0.5 0.5 0.5 1" />
        </body>
    """

    # Insert the obstacles
    xml_with_obstacles = xml_content[:insert_pos] + \
        obstacles + xml_content[insert_pos:]

    # Write to a new file
    with open("jackal_obstacles.xml", "w") as f:
        f.write(xml_with_obstacles)

    return "jackal_obstacles.xml"


def test_lidar_visualization():
    """Test the LiDAR visualization"""
    # Create an environment with obstacles
    xml_file = create_obstacle_env()

    # Create the environment with LiDAR
    env = Jackal_Env(
        xml_file=xml_file,
        render_mode="human",
        use_lidar=True,
        num_lidar_rays_h=360,
        num_lidar_rays_v=16,
        lidar_max_range=10.0
    )

    # Enable LiDAR visualization
    env.lidar_viz_enabled = True

    # Create a figure for top-down visualization
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))

    # Reset the environment
    observation, _ = env.reset()

    # Run a test loop
    for i in range(1000):
        # Take actions to move around and explore
        if i < 200:  # Go forward
            action = [0.5, 0.5]
        elif i < 400:  # Turn right
            action = [0.7, 0.1]
        elif i < 600:  # Go forward
            action = [0.5, 0.5]
        elif i < 800:  # Turn left
            action = [0.1, 0.7]
        else:  # Go forward
            action = [0.5, 0.5]

        # Step the environment
        observation, reward, terminated, truncated, info = env.step(action)

        # Visualize the LiDAR data in a top-down 2D view (updated every 10 steps)
        if i % 10 == 0:
            # Clear the plot
            ax.clear()

            # Get the LiDAR data for the middle vertical layer
            lidar_data = observation['lidar']
            middle_layer_idx = lidar_data.shape[0] // 2
            middle_layer = lidar_data[middle_layer_idx]

            # Convert polar coordinates to Cartesian for visualization
            angles = env.lidar.h_angles
            xs = middle_layer * np.cos(angles)
            ys = middle_layer * np.sin(angles)

            # Plot the robot and the LiDAR points
            ax.scatter(0, 0, s=100, c='blue', marker='o', label='Robot')
            ax.scatter(xs, ys, s=5, c='red', label='LiDAR Points')

            # Add robot heading indicator
            quat = observation['state'][3:7]  # Assuming quaternion is here
            # Convert quaternion to heading angle (simplified)
            # For proper conversion, use a quaternion library
            heading = 2 * np.arctan2(quat[3], quat[0])  # Very simplified!
            ax.arrow(0, 0, 0.5*np.cos(heading), 0.5*np.sin(heading),
                     head_width=0.1, head_length=0.2, fc='green', ec='green')

            # Set plot properties
            ax.set_xlim(-env.lidar.max_range, env.lidar.max_range)
            ax.set_ylim(-env.lidar.max_range, env.lidar.max_range)
            ax.set_aspect('equal')
            ax.grid(True)
            ax.set_title(f'LiDAR Top-Down View (Step {i})')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.legend()

            plt.draw()
            plt.pause(0.001)

        # Render the environment
        env.render()

        # Sleep to control simulation speed
        time.sleep(0.01)

        # Check if the episode is done
        if terminated or truncated:
            observation, _ = env.reset()

    env.close()


def test_point_cloud_saving():
    """Test saving LiDAR point clouds to file"""
    # Create the environment with LiDAR
    xml_file = create_obstacle_env()
    env = Jackal_Env(xml_file=xml_file, render_mode=None, use_lidar=True)

    # Reset the environment
    observation, _ = env.reset()

    # Take a few steps to get the robot in position
    for _ in range(10):
        observation, _, _, _, _ = env.step([0.5, 0.5])

    # Get the point cloud
    point_cloud = env.lidar.get_point_cloud()

    # Save to file in PLY format
    if len(point_cloud) > 0:
        with open('lidar_point_cloud.ply', 'w') as f:
            # Write header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(point_cloud)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")

            # Write points
            for point in point_cloud:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")

        print(
            f"Saved point cloud with {len(point_cloud)} points to lidar_point_cloud.ply")
    else:
        print("No valid points in point cloud")

    env.close()


if __name__ == "__main__":
    print("Testing LiDAR visualization...")
    test_lidar_visualization()

    print("Testing point cloud saving...")
    test_point_cloud_saving()
