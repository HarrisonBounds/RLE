import rclpy
from rclpy.node import Node
from RLE_interfaces.msg import Action

class ActionPublisher(Node):

    def __init__(self):
        super.__init__("action_publisher")
        self.action_pub = rclpy.publisher(Action, 'action', 10)

def main(args=None):
    """Entrypoint for the waypoint ROS node."""
    rclpy.init(args=args)
    node = ActionPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    import sys
    main(sys.argv)