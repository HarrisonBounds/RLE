from urdf2mjcf import run

run(
    urdf_path="path/to/your/robot.urdf",
    mjcf_path="path/to/save/robot.mjcf",
    copy_meshes=True,
)
