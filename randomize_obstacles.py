import xml.etree.ElementTree as ET
import random

# Tags in XML to identify where to insert generated obstacles
START_TAG = "<!--START_OBSTACLES-->"
END_TAG = "<!--END_OBSTACLES-->"

# Template for an obstacle in the XML file
OBSTACLE_TEMPLATE = (
    "<body name=\"obstacle{obstID}\" pos=\"{pX} {pY} {pZ}\">"
    "<geom type=\"{obstType}\" size=\"{size}\" material=\"obstacleColor\" />"
    "</body>\n"
)

# Load the XML file
FILE_PATH = "jackal_obstacles.xml"

min_box_size = 0.1
max_box_size = 2.0
min_cylinder_radius = 0.1
max_cylinder_radius = 1.0
min_cylinder_height = 0.1
max_cylinder_height = 2.0
min_sphere_radius = 0.1
max_sphere_radius = 1.0


def generate_random_size(obst_type: str) -> str:
    precision = 2  # Number of decimal places for size values

    def fmt(val):
        return f"{val:.{precision}f}"

    if obst_type == "box":
        # size is a string: "L W H"
        L = random.uniform(min_box_size, max_box_size)
        W = random.uniform(min_box_size, max_box_size)
        H = random.uniform(min_box_size, max_box_size)
        # Ensure L >= W >= H for consistency
        if L < W:
            L, W = W, L
        if W < H:
            W, H = H, W
        if L < H:
            L, H = H, L
        return f"{fmt(L)} {fmt(W)} {fmt(H)}"
    elif obst_type == "cylinder":
        # Size is a string: "radius height"
        R = random.uniform(min_cylinder_radius, max_cylinder_radius)
        H = random.uniform(min_cylinder_height, max_cylinder_height)
        return f"{fmt(R)} {fmt(H)}"
    elif obst_type == "sphere":
        # size is a string: "radius"
        R = random.uniform(min_sphere_radius, max_sphere_radius)
        return f"{fmt(R)}"
    else:
        raise ValueError("Invalid obstacle type")


def generate_random_position(area_size: tuple, type: str, size: str) -> tuple:
    pass


def create_xml_obstacle(obstacle_id, position, obstType, size) -> str:
    # Validate inputs
    assert len(position) == 3, "Position must be a tuple of (x, y, z)"
    assert obstType in ["box", "cylinder",
                        "sphere"], "Invalid obstacle type"
    # Create string and return
    return OBSTACLE_TEMPLATE.format(
        obstID=obstacle_id,
        pX=position[0],
        pY=position[1],
        pZ=position[2],
        obstType=obstType,
        size=size
    )


def generate_random_obstacles(num_obstacles: int, area_size: tuple) -> list[str]:
    # Given a number of obstacles and the bounding area size,
    # generate a list of random obstacles with unique Ids
    # Ensure that no obstacles overlap with each other
    # Ensure that all obstacles are within the area bounds and don't intersect the walls
    # Ensure that no obstacles are generated at the robot's position (0, 0, 0)
    assert len(
        area_size) == 4, "Area size must be a tuple of (minX, minY, maxX, maxY)"
    assert num_obstacles > 0, "Number of obstacles must be greater than 0"
    obstacles = []
    for i in range(num_obstacles):
        obst_id = i + 1
        obst_type = random.choice(["box", "cylinder", "sphere"])
        size = generate_random_size(obst_type)
        pos = generate_random_position(area_size, obst_type, size)


def insert_obstacles(obstacles: list[str], xml_tree: ET.ElementTree):
    start_index = xml_tree.find(START_TAG)
    end_index = xml_tree.find(END_TAG)
    if start_index == -1 or end_index == -1:
        raise ValueError("Start or end tags not found in the XML file.")
    # Insert obstacles between the start and end tags
    obstacles_str = "\n".join(obstacles)
    xml_tree.text = (
        xml_tree.text[:start_index + len(START_TAG)] +
        obstacles_str +
        xml_tree.text[end_index:]
    )


if __name__ == "__main__":
    XML_TREE = ET.parse(FILE_PATH).getroot()
    print(generate_random_size("box"))
    print(generate_random_size("cylinder"))
    print(generate_random_size("sphere"))
