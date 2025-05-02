import mujoco
from mujoco import mjcf_from_urdf

mjcf_model = mjcf_from_urdf.from_path("jackal_description/urdf/jackal.urdf")
mjcf_model.save("jackal_converted.xml")
