#!/usr/bin/env python

import rospy
import sys

import numpy as np
import cupy as cp
import ros_numpy
import matplotlib.pyplot as plt
from skimage.io import imshow

import message_filters
from sensor_msgs.msg import Image, CameraInfo
from sensor_msgs.msg import PointCloud2, Image
from cv_bridge import CvBridge

from semantic_pointcloud.semantic_segmentation_parameters import SemanticSegmentationParameter
from semantic_pointcloud.networks import resolve_model
from semantic_pointcloud.utils import decode_max


class SemanticSegmentationNode:
    def __init__(self, sensor_name):
        """Get parameter from server, initialize variables and semantics, register publishers and subscribers.

        Args:
            sensor_name (str): Name of the sensor in the ros param server.
        """
        # TODO: if this is going to be loaded from another package we might need to change namespace
        self.param: SemanticSegmentationParameter = SemanticSegmentationParameter()
        self.param.feature_config.input_size = [80, 160]
        namesp = rospy.get_name()
        if rospy.has_param(namesp + "/subscribers"):
            config = rospy.get_param(namesp + "/subscribers")
            self.param: SemanticSegmentationParameter = SemanticSegmentationParameter.from_dict(config[sensor_name])
        else:
            print("NO ROS ENV found.")

        print("--------------Pointcloud Parameters-------------------")
        print(self.param.dumps_yaml())
        print("--------------End of Parameters-----------------------")
        self.semseg_color_map = None
        # setup custom dtype
        # setup semantics
        self.feature_extractor = None
        self.semantic_model = None
        self.initialize_semantics()

        # setup pointcloud creation
        self.cv_bridge = CvBridge()
        self.P = None
        self.header = None
        self.register_sub_pub()
        self.prediction_img = None

    def initialize_semantics(self):
        if self.param.semantic_segmentation:
            self.semantic_model = resolve_model(self.param.segmentation_model, self.param)

        if self.param.feature_extractor:
            self.feature_extractor = resolve_model(self.param.feature_config.name, self.param.feature_config)

    def register_sub_pub(self):
        """Register publishers and subscribers."""
        # subscribers
        rospy.Subscriber(self.param.image_topic, Image, self.image_callback)

        # publishers
        if self.param.semantic_segmentation:
            self.seg_pub = rospy.Publisher(self.param.sem_seg_topic, Image, queue_size=2)
            self.seg_im_pub = rospy.Publisher(self.param.sem_seg_image_topic, Image, queue_size=2)
            self.semseg_color_map = self.color_map(len(self.param.channels))
            if self.param.show_label_legend:
                self.color_map_viz()
        if self.param.feature_extractor:
            self.feature_pub = rospy.Publisher(self.param.feature_topic, Image, queue_size=2)

    def color_map(self, N=256, normalized=False):
        """Create a color map for the class labels.

        Args:
            N (int):
            normalized (bool):
        """

        def bitget(byteval, idx):
            return (byteval & (1 << idx)) != 0

        dtype = "float32" if normalized else "uint8"
        cmap = np.zeros((N + 1, 3), dtype=dtype)
        for i in range(N + 1):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7 - j)
                g = g | (bitget(c, 1) << 7 - j)
                b = b | (bitget(c, 2) << 7 - j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

        cmap = cmap / 255 if normalized else cmap
        return cmap[1:]

    def color_map_viz(self):
        """Display the color map of the classes."""
        nclasses = len(self.param.channels)
        row_size = 50
        col_size = 500
        if self.param.semantic_segmentation:
            cmap = self.semseg_color_map
        array = np.empty((row_size * (nclasses), col_size, cmap.shape[1]), dtype=cmap.dtype)
        for i in range(nclasses):
            array[i * row_size : i * row_size + row_size, :] = cmap[i]
        imshow(array)
        plt.yticks([row_size * i + row_size / 2 for i in range(nclasses)], self.param.channels)
        plt.xticks([])
        plt.show()

    def image_callback(self, rgb_msg):
        image = cp.asarray(self.cv_bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8"))
        self.header = rgb_msg.header
        self.process_image(image)

        if self.param.semantic_segmentation:
            self.publish_segmentation()
            self.publish_segmentation_image()
        # if self.param.feature_extractor:
        #     self.publish_feature_image()

    def process_image(self, image):
        """Depending on setting generate color, semantic segmentation or feature channels.

        Args:
            image:
            u:
            v:
            points:
        """

        if self.param.semantic_segmentation:
            self.sem_seg = self.semantic_model["model"](image)

        # if self.feature_channels is not None:
        #             prediction = self.feature_extractor["model"](image)

    def publish_segmentation(self):
        probabilities = self.sem_seg
        img = probabilities.get()
        img = np.transpose(img, (1, 2, 0)).astype(np.float32)
        seg_msg = self.cv_bridge.cv2_to_imgmsg(img, encoding="passthrough")
        seg_msg.header.frame_id = self.header.frame_id
        seg_msg.header.stamp = self.header.stamp
        self.seg_pub.publish(seg_msg)

    def publish_segmentation_image(self):
        colors = None
        probabilities = self.sem_seg
        if self.param.semantic_segmentation:
            colors = cp.asarray(self.semseg_color_map)
            assert colors.ndim == 2 and colors.shape[1] == 3

        # prob = cp.zeros((len(self.param.channels),) + probabilities.shape[1:])
        # if "class_max" in self.param.fusion:
        #     # decode, create an array with all possible classes and insert probabilities
        #     it = 0
        #     for iit, (chan, fuse) in enumerate(zip(self.param.channels, self.param.fusion)):
        #         if fuse in ["class_max"]:
        #             temp = probabilities[it]
        #             temp_p, temp_i = decode_max(temp)
        #             temp_i.choose(prob)
        #             c = cp.mgrid[0 : temp_i.shape[0], 0 : temp_i.shape[1]]
        #             prob[temp_i, c[0], c[1]] = temp_p
        #             it += 1
        #         elif fuse in ["class_bayesian", "class_average"]:
        #             # assign fixed probability to correct index
        #             if chan in self.semantic_model["model"].segmentation_channels:
        #                 prob[self.semantic_model["model"].segmentation_channels[chan]] = probabilities[it]
        #                 it += 1
        #     img = cp.argmax(prob, axis=0)
        #
        # else:
        img = cp.argmax(probabilities, axis=0)
        img = colors[img].astype(cp.uint8)  # N x H x W x 3
        img = img.get()
        seg_msg = self.cv_bridge.cv2_to_imgmsg(img, encoding="rgb8")
        seg_msg.header.frame_id = self.header.frame_id
        seg_msg.header.stamp = self.header.stamp
        self.seg_im_pub.publish(seg_msg)


if __name__ == "__main__":
    arg = sys.argv[1]
    sensor_name = arg
    rospy.init_node("semantic_segmentation_node", anonymous=True, log_level=rospy.INFO)
    node = SemanticSegmentationNode(sensor_name)
    rospy.spin()
