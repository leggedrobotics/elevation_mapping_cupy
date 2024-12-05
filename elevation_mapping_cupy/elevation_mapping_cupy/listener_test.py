import rclpy
from rclpy.node import Node
import tf2_ros
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage

class TfListener(Node):
    def __init__(self):
        super().__init__('tf_listener')
        self.subscription = self.create_subscription(
            TFMessage,
            '/tf',
            self.listener_callback,
            10)
        self.last_time = None

    def listener_callback(self, msg):
        for transform in msg.transforms:
            current_time = transform.header.stamp.sec + transform.header.stamp.nanosec * 1e-9
            if self.last_time is not None:
                delta = current_time - self.last_time
                self.get_logger().info(f"Delta time: {delta:.6f} seconds")
            self.last_time = current_time

def main(args=None):
    rclpy.init(args=args)
    node = TfListener()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
