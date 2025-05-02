from urdf2mjcf import run

run(
    urdf_path="jackal_description/urdf/jackal.urdf",
    mjcf_path="jackal.xml",
    copy_meshes=True,
)
