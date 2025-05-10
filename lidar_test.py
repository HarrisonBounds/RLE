# lidar_test.py
import numpy as np
import mujoco
import mujoco.viewer
import time

# Import the VLP16 sensor
from lidar_sensor import VLP16Sensor


def test_vlp16_lidar():
    """Test the VLP-16 LiDAR simulation with the Jackal robot in an obstacle environment"""
    print("Loading MuJoCo model...")
    model = mujoco.MjModel.from_xml_path("jackal_obstacles.xml")
    data = mujoco.MjData(model)

    # Reset the simulation
    mujoco.mj_resetData(model, data)

    # Initialize viewer
    print("Initializing viewer...")
    viewer = mujoco.viewer.launch_passive(model, data)

    # Initialize the VLP-16 sensor
    print("Initializing VLP-16 sensor...")
    lidar = VLP16Sensor(
        model=model,
        data=data,
        lidar_name="velodyne",
        horizontal_resolution=0.4,  # degrees
        rotation_rate=10,           # Hz
        max_range=100.0            # meters
    )

    print("\nStarting simulation...")
    print("Press Ctrl+C to stop the simulation\n")

    # Run the simulation
    start_time = time.time()
    last_print_time = start_time

    try:
        for i in range(1000):
            # Set control signals BEFORE stepping
            if i < 200:
                # Move forward
                ctrl = [0.5, 0.5]  # [left, right] wheel velocities
            elif i < 400:
                # Turn right
                ctrl = [0.7, 0.1]
            elif i < 600:
                # Move forward
                ctrl = [0.5, 0.5]
            elif i < 800:
                # Turn left
                ctrl = [0.1, 0.7]
            else:
                # Move forward
                ctrl = [0.5, 0.5]

            # Apply control signals
            data.ctrl[0] = ctrl[0]  # Left wheels
            data.ctrl[1] = ctrl[1]  # Right wheels

            # Step the simulation AFTER setting controls
            mujoco.mj_step(model, data)

            # CRITICAL: Update the viewer to show movement
            if viewer.is_running():
                viewer.sync()
            else:
                break  # Exit if viewer is closed

            # Update the LiDAR (casting all rays each frame)
            scan_result = lidar.update()

            # Print data periodically (every 1 second)
            current_time = time.time()
            if current_time - last_print_time > 1.0:
                # Get scan statistics
                stats = lidar.get_statistics()

                # Print information
                print(f"\n--- Step {i} ---")
                print(f"Robot position: {data.qpos[:3]}")
                # Added velocity info
                print(f"Robot velocity: {data.qvel[:3]}")
                print(f"Control signals: left={ctrl[0]}, right={ctrl[1]}")
                print(f"LiDAR position: {scan_result['lidar_position']}")
                print(f"Number of rays: {stats['total_rays']}")
                print(
                    f"Valid points: {stats['valid_points']} ({stats['hit_percentage']:.2f}%)")

                if stats['valid_points'] > 0:
                    print(f"Range statistics: min={stats['min_distance']:.2f}m, " +
                          f"max={stats['max_distance']:.2f}m, " +
                          f"mean={stats['mean_distance']:.2f}m")

                    # Print a few sample points from the pointcloud
                    points = scan_result['points']
                    if len(points) > 0:
                        num_samples = min(5, len(points))
                        print("\nSample points from pointcloud:")
                        for j in range(num_samples):
                            print(
                                f"  Point {j+1}: ({points[j][0]:.2f}, {points[j][1]:.2f}, {points[j][2]:.2f})")

                last_print_time = current_time

            # Save a point cloud snapshot at step 500
            if i == 500:
                lidar.save_point_cloud_ply("vlp16_point_cloud.ply")
                print("\nSaved point cloud to vlp16_point_cloud.ply")

            # Control simulation speed (sleep to maintain real-time factor)
            time_to_sleep = max(0, model.opt.timestep -
                                (time.time() - start_time))
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)
            start_time = time.time()

    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
    finally:
        # Make sure to close the viewer
        viewer.close()

    print("\nSimulation complete")


if __name__ == "__main__":
    test_vlp16_lidar()
