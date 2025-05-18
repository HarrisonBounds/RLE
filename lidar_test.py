# async_lidar_test.py
import asyncio
import mujoco
import mujoco.viewer
import time
import numpy as np
import matplotlib.pyplot as plt

from lidar_sensor import VLP16Sensor


async def main():
    """Test the asynchronous VLP-16 LiDAR sensor with the Jackal robot"""
    print("Loading MuJoCo model...")
    model = mujoco.MjModel.from_xml_path("jackal_obstacles.xml")
    data = mujoco.MjData(model)

    # Reset the simulation
    mujoco.mj_resetData(model, data)

    # Initialize viewer (non-blocking)
    print("Initializing viewer...")
    viewer = mujoco.viewer.launch_passive(model, data)

    # Initialize the VLP-16 sensor
    print("Initializing VLP-16 sensor...")
    lidar = VLP16Sensor(
        model=model,
        data=data,
        lidar_name="velodyne",
        horizontal_resolution=1.0,  # degrees
        rotation_rate=10,          # Hz
        max_range=100.0            # meters
    )

    # Enable matplotlib to work with the event loop
    plt.ion()  # Turn on interactive mode

    # Start the LiDAR in async mode
    await lidar.start_async(visualize=True, update_rate=10, visualization_rate=10)

    print("\nStarting simulation...")
    print("Press Ctrl+C to stop the simulation\n")

    # Define the duration and control patterns
    sim_duration = 30  # seconds
    sim_step = 0.002  # simulation timestep
    start_time = time.time()

    try:
        while time.time() - start_time < sim_duration and viewer.is_running():
            # Calculate elapsed sim time
            elapsed = time.time() - start_time

            # Set control signals based on time
            if elapsed < 5:
                # Move forward
                ctrl = [0.5, 0.5]  # [left, right] wheel velocities
            elif elapsed < 10:
                # Turn right
                ctrl = [0.7, 0.1]
            elif elapsed < 15:
                # Move forward
                ctrl = [0.5, 0.5]
            elif elapsed < 20:
                # Turn left
                ctrl = [0.1, 0.7]
            else:
                # Move forward
                ctrl = [0.5, 0.5]

            # Apply control signals
            data.ctrl[0] = ctrl[0]  # Left wheels
            data.ctrl[1] = ctrl[1]  # Right wheels

            # Step the simulation
            mujoco.mj_step(model, data)

            # Update the viewer
            if viewer.is_running():
                viewer.sync()
            else:
                break

            # Control simulation speed
            await asyncio.sleep(sim_step)

            # Process any GUI events for matplotlib
            plt.pause(0.0001)  # Small pause to let matplotlib process events

    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
    finally:
        # Stop the LiDAR async tasks
        await lidar.stop_async()

        # Close the viewer
        viewer.close()

        # Close all matplotlib windows
        plt.close('all')

    print("\nSimulation complete")


if __name__ == "__main__":
    # Run the main coroutine
    asyncio.run(main())
