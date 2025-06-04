import xml.etree.ElementTree as ET
import random
import time
import numpy as np

# Tags in XML to identify where to insert generated obstacles
START_TAG = "<!--START_OBSTACLES-->"
END_TAG = "<!--END_OBSTACLES-->"
GOAL_START_TAG = "<!--START_GOAL-->"
GOAL_END_TAG = "<!--END_GOAL-->"

# Template for an obstacle in the XML file
OBSTACLE_TEMPLATE = (
    "\t\t<body name=\"obstacle{obstID}\" pos=\"{pX:.3f} {pY:.3f} {pZ:.3f}\">\n"
    "\t\t\t<geom type=\"{obstType}\" size=\"{size}\" material=\"obstacleColor\" group=\"1\"/>\n"
    "\t\t</body>"
)

GOAL_TEMPLATE = (
    "\t\t<body name=\"goal\" pos=\"{pX:.3f} {pY:.3f} 0.125\" euler=\"0 0 {yaw:.3f}\">\n"
    "\t\t\t<geom name=\"goal_geom\" type=\"box\" size=\"0.25 0.25 0.25\" material=\"green\" group=\"2\"/>\n"
    "\t\t</body>"
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


class Obstacle:
    def __init__(self, obst_id: int, obst_type: str, size: str, position: tuple):
        self.obst_id = obst_id
        self.obst_type = obst_type
        self.size = size
        self.position = position

    def get_largest_bounding_box(self) -> tuple:
        if self.obst_type == "box":
            # For a box, the bounding box is the same as its size
            L, W, H = [float(s) for s in self.size.split()]
            return (self.position[0], self.position[1], self.position[2],
                    self.position[0] + L, self.position[1] + W, self.position[2] + H)
        elif self.obst_type == "cylinder":
            # For a cylinder, the bounding box is defined by its radius and height
            R, H = [float(s) for s in self.size.split()]
            return (self.position[0] - R, self.position[1] - R, self.position[2],
                    self.position[0] + R, self.position[1] + R, self.position[2] + H)
        elif self.obst_type == "sphere":
            # For a sphere, the bounding box is defined by its radius
            R = float(self.size.split()[0])
            return (self.position[0] - R, self.position[1] - R, self.position[2] - R,
                    self.position[0] + R, self.position[1] + R, self.position[2] + R)
        else:
            raise ValueError("Invalid obstacle type")

    def intersects(self, other: "Obstacle") -> bool:
        assert isinstance(
            other, Obstacle), "Other must be an instance of Obstacle"
        size = self.size.split(" ")
        if self.obst_type == "box":
            size = [float(s) for s in size]
            L, W, H = size
        elif self.obst_type == "cylinder":
            size = [float(s) for s in size]
            R, H = size
        elif self.obst_type == "sphere":
            size = [float(s) for s in size]
            R = size[0]
        other_size = other.size.split(" ")
        if other.obst_type == "box":
            other_size = [float(s) for s in other_size]
            other_L, other_W, other_H = other_size
        elif other.obst_type == "cylinder":
            other_size = [float(s) for s in other_size]
            other_R, other_H = other_size
        elif other.obst_type == "sphere":
            other_size = [float(s) for s in other_size]
            other_R = other_size[0]
        # Intersection logic
        if self.obst_type == "box" and other.obst_type == "box":
            # Check if two boxes intersect
            return not (self.position[0] + L < other.position[0] or
                        self.position[0] > other.position[0] + other_L or
                        self.position[1] + W < other.position[1] or
                        self.position[1] > other.position[1] + other_W or
                        self.position[2] + H < other.position[2] or
                        self.position[2] > other.position[2] + other_H)
        elif self.obst_type == "cylinder" and other.obst_type == "cylinder":
            # Check if two cylinders intersect (using 2D circle overlap and height)
            dx = self.position[0] - other.position[0]
            dy = self.position[1] - other.position[1]
            dist_2d = (dx ** 2 + dy ** 2) ** 0.5
            overlap_2d = dist_2d < (R + other_R)
            z1_min = self.position[2] - H / 2
            z1_max = self.position[2] + H / 2
            z2_min = other.position[2] - other_H / 2
            z2_max = other.position[2] + other_H / 2
            overlap_z = not (z1_max < z2_min or z1_min > z2_max)
            return overlap_2d and overlap_z
        elif self.obst_type == "sphere" and other.obst_type == "sphere":
            # Check if two spheres intersect
            distance = ((self.position[0] - other.position[0]) ** 2 +
                        (self.position[1] - other.position[1]) ** 2 +
                        (self.position[2] - other.position[2]) ** 2) ** 0.5
            return distance < (R + other_R)
        else:
            # For mixed types, we can use a simple bounding box check
            self_bbox = self.get_largest_bounding_box()
            other_bbox = other.get_largest_bounding_box()
            return not (self_bbox[3] < other_bbox[0] or
                        self_bbox[0] > other_bbox[3] or
                        self_bbox[4] < other_bbox[1] or
                        self_bbox[1] > other_bbox[4] or
                        self_bbox[5] < other_bbox[2] or
                        self_bbox[2] > other_bbox[5])


def generate_random_size(obst_type: str) -> str:
    precision = 2  # Number of decimal places for size values

    def fmt(val):
        return f"{val:.{precision}f}"

    if obst_type == "box":
        # size is a string: "L W H"
        L = random.uniform(min_box_size, max_box_size)
        W = random.uniform(min_box_size, max_box_size)
        H = random.uniform(min_box_size, max_box_size)
        # Ensure L >= W >= H
        L, W, H = sorted([L, W, H], reverse=True)
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


def generate_random_position(area_size: tuple, type: str, size: str, obstacles: list) -> tuple:
    # Area size is a tuple of (minX, minY, maxX, maxY)
    assert len(
        area_size) == 4, "Area size must be a tuple of (minX, minY, maxX, maxY)"
    assert type in ["box", "cylinder", "sphere"], "Invalid obstacle type"
    # Parse size depending on type
    if type == "box":
        dim = [float(s) for s in size.split(" ")]
        assert len(dim) == 3, "Box size must be a string of 'L W H'"
    elif type == "cylinder":
        dim = [float(s) for s in size.split(" ")]
        assert len(dim) == 2, "Cylinder size must be a string of 'radius height'"
    elif type == "sphere":
        dim = [float(s) for s in size.split(" ")]
        assert len(dim) == 1, "Sphere size must be a string of 'radius'"
    # Generate random position within the area bounds that does not intersect with any other
    # obstacles
     # Robot's position is at (0, 0, 0), define a 2m area around it to avoid
    min_x, min_y, max_x, max_y = area_size
    robot_area = 2  # sphere radius around the robot to avoid [0 -> 2]
    while True:
        x = random.uniform(min_x, max_x)
        y = random.uniform(min_y, max_y)
        # Assign the Z such that the obstacle is resting on the ground
        z = 0.0
        if type == "box":
            # For a box, the position is the center of the base
            z = dim[2] / 2  # Height of the box
            pos = (x - dim[0] / 2, y - dim[1] / 2, z)
        elif type == "cylinder":
            # For a cylinder, the position is the center of the base
            z = dim[1] / 2  # Height of the cylinder
            pos = (x, y, z)
        elif type == "sphere":
            # For a sphere, the position is the center
            z = dim[0]  # Radius of the sphere
            pos = (x, y, z)
        else:
            raise ValueError("Invalid obstacle type")
        # Check if the position is within the robot's area
        if (x ** 2 + y ** 2) < robot_area ** 2:
            continue
        # Check if the position intersects with any existing obstacles
        intersects = False
        for obst in obstacles:
            if obst.intersects(Obstacle(-1, type, size, pos)):
                intersects = True
                break
        if not intersects:
            return pos


def generate_random_goal(area_size, obstacles: list):
    min_x, min_y, max_x, max_y = area_size
    # Generate a random X and Y position for the goal that doesn't intersect with any other
    # obstacles and stays away from the robot area

    def in_robot_area(x, y, robot_area=2) -> bool:
        return (x**2 + y**2) < robot_area**2

    def intersects_with_obstacles(x, y, obstacles):
        # Check if the goal position intersects with any existing obstacles
        for obst in obstacles:
            if obst.intersects(Obstacle(-1, "box", "0.25 0.25 0.25", (x, y, 0.125))):
                return True
        return False

    random_x = random.uniform(min_x, max_x)
    random_y = random.uniform(min_y, max_y)
    random_yaw = random.uniform(-np.pi, np.pi)  # Random yaw in radians
    while True:
        if in_robot_area(random_x, random_y) or intersects_with_obstacles(random_x, random_y, obstacles):
            random_x = random.uniform(min_x, max_x)
            random_y = random.uniform(min_y, max_y)
        else:
            return (random_x, random_y, random_yaw)


def generate_random_obstacles(num_obstacles: int, area_size: tuple) -> list[Obstacle]:
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
        pos = generate_random_position(area_size, obst_type, size, obstacles)
        obstacles.append(
            Obstacle(obst_id, obst_type, size, pos)
        )
    return obstacles


def insert_obstacles_raw(obstacles, in_path, out_path):
    # 1) Read the original XML
    with open(in_path, 'r', encoding='utf-8') as f:
        xml = f.read()

    # 2) Build the block of obstacle‐strings
    obstacle_lines = [
        OBSTACLE_TEMPLATE.format(
            obstID=o.obst_id,
            pX=o.position[0],
            pY=o.position[1],
            pZ=o.position[2],
            obstType=o.obst_type,
            size=o.size
        )
        for o in obstacles
    ]
    block = "\n".join(obstacle_lines)

    # 3) Split on the markers
    try:
        before, rest = xml.split(START_TAG, 1)
        _, after = rest.split(END_TAG, 1)
    except ValueError:
        raise ValueError(
            "Could not find both START_OBSTACLES and END_OBSTACLES in the file")

    # 4) Reassemble with your raw block in between
    new_xml = (
        before
        + START_TAG
        + "\n"       # optional: put block on its own lines
        + block
        + "\n"
        + "\t\t" + END_TAG
        + after
    )

    # 5) Write it out
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(new_xml)


def insert_goal_raw(goal_pose, in_path, out_path):
    # 1) Read the original XML
    with open(in_path, 'r', encoding='utf-8') as f:
        xml = f.read()

    # 2) Build the goal string
    goal_string = GOAL_TEMPLATE.format(
        pX=goal_pose[0],
        pY=goal_pose[1],
        yaw=goal_pose[2]
    )

    # 3) Split on the markers
    try:
        before, rest = xml.split(GOAL_START_TAG, 1)
        _, after = rest.split(GOAL_END_TAG, 1)
    except ValueError:
        raise ValueError(
            "Could not find both START_GOAL and END_GOAL in the file")

    # 4) Reassemble with your goal string in between
    new_xml = (
        before
        + GOAL_START_TAG
        + "\n"
        + goal_string
        + "\n"
        + "\t\t" + GOAL_END_TAG
        + after
    )

    # 5) Write it out
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(new_xml)


def randomize_environment(env_path: str, min_num_obstacles: int = 3, max_num_obstacles: int = 7) -> None:
    """
    Randomizes the environment by generating a random number of obstacles
    and inserting them into the specified XML file.

    :param env_path: Path to the XML environment file.
    :param min_num_obstacles: Minimum number of obstacles to generate.
    :param max_num_obstacles: Maximum number of obstacles to generate.
    """
    num_obstacles = random.randint(min_num_obstacles, max_num_obstacles)
    area_size = (-5, -5, 5, 5)  # (minX, minY, maxX, maxY)
    random_obstacles = generate_random_obstacles(num_obstacles, area_size)
    random_goal = generate_random_goal(area_size, random_obstacles)
    insert_obstacles_raw(
        random_obstacles,
        in_path=env_path,
        out_path=env_path
    )
    insert_goal_raw(
        random_goal,
        in_path=env_path,
        out_path=env_path
    )


if __name__ == "__main__":
    REPETITIONS = 1
    avg_runtime = 0.0
    for _ in range(REPETITIONS):
        start_time = time.perf_counter()
        NUM_OBSTACLES = 10
        AREA_SIZE = (-5, -5, 5, 5)  # (minX, minY, maxX, maxY)
        random_obstacles = generate_random_obstacles(NUM_OBSTACLES, AREA_SIZE)
        random_goal = generate_random_goal(AREA_SIZE, random_obstacles)
        NEW_FILE_PATH = "jackal_obstacles_randomized.xml"
        insert_obstacles_raw(
            random_obstacles,
            in_path=FILE_PATH,
            out_path=NEW_FILE_PATH
        )
        insert_goal_raw(
            random_goal,
            in_path=FILE_PATH,
            out_path=NEW_FILE_PATH
        )
        elapsed = (time.perf_counter() - start_time) * 1000  # milliseconds
        avg_runtime += elapsed
    avg_runtime /= REPETITIONS
    print(
        f"Generated {NUM_OBSTACLES} random obstacles and saved to {NEW_FILE_PATH}"
    )
    print(
        f"Average elapsed time over {REPETITIONS} repetitions: {avg_runtime:.3f} ms")
