import rospy
import sys

import numpy as np
import ros_numpy
import matplotlib.pyplot as plt
from skimage.io import imshow

import message_filters
from sensor_msgs.msg import Image, CameraInfo
from sensor_msgs.msg import PointCloud2, Image
from cv_bridge import CvBridge


class ImageDebugNode:
    def __init__(self):
        # setup pointcloud creation
        self.cv_bridge = CvBridge()
        # rospy.Subscriber("", CameraInfo, self.cam_info_callback)
        rospy.Subscriber("/zed2i/zed_node/right/image_rect_color", Image, self.image_callback)
        self.debug_pub = rospy.Publisher("/debug_image", Image, queue_size=2)

    def image_callback(self, rgb_msg):
        img = self.cv_bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
        out = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint16)
        out[:, : int(img.shape[1] / 2)] = 1
        seg_msg = self.cv_bridge.cv2_to_imgmsg(out, encoding="mono16")
        seg_msg.header = rgb_msg.header
        self.debug_pub.publish(seg_msg)
        # seg_msg.header.frame_id = self.header.frame_id
        # self.seg_pub.publish(seg_msg)


if __name__ == "__main__":
    rospy.init_node("image_debug_node", anonymous=True, log_level=rospy.INFO)
    print("Start Debug Image Node")
    node = ImageDebugNode()
    rospy.spin()
