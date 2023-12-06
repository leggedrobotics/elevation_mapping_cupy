#!/usr/bin/env python

import rospy
import sys

import numpy as np
import cupy as cp
import cv2

np.float = np.float64  # temp fix for following import suggested at https://github.com/eric-wieser/ros_numpy/issues/37
np.bool = np.bool_
import ros_numpy
import matplotlib.pyplot as plt
from skimage.io import imshow

import message_filters
from sensor_msgs.msg import Image, CameraInfo
from sensor_msgs.msg import PointCloud2, Image, CompressedImage
from cv_bridge import CvBridge

from semantic_sensor.image_parameters import ImageParameter
from semantic_sensor.networks import resolve_model
from sklearn.decomposition import PCA

from elevation_map_msgs.msg import ChannelInfo


class SemanticSegmentationNode:
    def __init__(self, sensor_name):
        """Get parameter from server, initialize variables and semantics, register publishers and subscribers.

        Args:
            sensor_name (str): Name of the sensor in the ros param server.
        """
        # TODO: if this is going to be loaded from another package we might need to change namespace
        self.param: ImageParameter = ImageParameter()
        self.param.feature_config.input_size = [80, 160]
        namespace = rospy.get_name()
        if rospy.has_param(namespace):
            config = rospy.get_param(namespace)
            self.param: ImageParameter = ImageParameter.from_dict(config[sensor_name])
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

        node_name = rospy.get_name()
        # subscribers
        if self.param.camera_info_topic is not None and self.param.resize is not None:
            rospy.Subscriber(self.param.camera_info_topic, CameraInfo, self.image_info_callback)
            self.feat_im_info_pub = rospy.Publisher(
                node_name + "/" + self.param.camera_info_topic + "_resized", CameraInfo, queue_size=2
            )

        if "compressed" in self.param.image_topic:
            self.compressed = True
            self.subscriber = rospy.Subscriber(
                self.param.image_topic, CompressedImage, self.image_callback, queue_size=2
            )
        else:
            self.compressed = False
            rospy.Subscriber(self.param.image_topic, Image, self.image_callback)

        # publishers
        if self.param.semantic_segmentation:
            self.seg_pub = rospy.Publisher(node_name + "/" + self.param.publish_topic, Image, queue_size=2)
            self.seg_im_pub = rospy.Publisher(node_name + "/" + self.param.publish_image_topic, Image, queue_size=2)
            self.semseg_color_map = self.color_map(len(self.param.channels))
            if self.param.show_label_legend:
                self.color_map_viz()
        if self.param.feature_extractor:
            self.feature_pub = rospy.Publisher(node_name + "/" + self.param.feature_topic, Image, queue_size=2)
            self.feat_im_pub = rospy.Publisher(node_name + "/" + self.param.feat_image_topic, Image, queue_size=2)
            self.feat_channel_info_pub = rospy.Publisher(
                node_name + "/" + self.param.feat_channel_info_topic, ChannelInfo, queue_size=2
            )

        self.channel_info_pub = rospy.Publisher(
            node_name + "/" + self.param.channel_info_topic, ChannelInfo, queue_size=2
        )

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
        cmap[1] = np.array([81, 113, 162])
        cmap[2] = np.array([81, 113, 162])
        cmap[3] = np.array([188, 63, 59])
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

    def image_info_callback(self, msg):
        """Callback for camera info.

        Args:
            msg (CameraInfo):
        """
        self.P = np.array(msg.P).reshape(3, 4)
        self.height = int(self.param.resize * msg.height)
        self.width = int(self.param.resize * msg.width)
        self.info = msg
        self.info.height = self.height
        self.info.width = self.width
        self.P = np.array(msg.P).reshape(3, 4)
        self.P[:2, :3] = self.P[:2, :3] * self.param.resize
        self.info.K = self.P[:3, :3].flatten().tolist()
        self.info.P = self.P.flatten().tolist()

    def image_callback(self, rgb_msg):
        if self.P is None:
            return
        if self.compressed:
            image = self.cv_bridge.compressed_imgmsg_to_cv2(rgb_msg)
            if self.param.resize is not None:
                image = cv2.resize(image, dsize=(self.width, self.height))
            image = cp.asarray(image)
        else:
            image = self.cv_bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
            if self.param.resize is not None:
                image = cv2.resize(image, dsize=(self.width, self.height))
            image = cp.asarray(image)
        self.header = rgb_msg.header
        self.process_image(image)

        if self.param.semantic_segmentation:
            self.publish_segmentation()
            self.publish_segmentation_image()
            self.publish_channel_info([f"sem_{c}" for c in self.param.channels], self.channel_info_pub)
        if self.param.feature_extractor:
            self.publish_feature()
            self.publish_feature_image(self.features)
            self.publish_channel_info([f"feat_{i}" for i in range(self.features.shape[0])], self.feat_channel_info_pub)
        if self.param.resize is not None:
            self.pub_info()

    def pub_info(self):
        self.feat_im_info_pub.publish(self.info)

    def publish_channel_info(self, channels, pub):
        """Publish fusion info."""
        info = ChannelInfo()
        info.header = self.header
        info.channels = channels
        pub.publish(info)

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

        if self.param.feature_extractor:
            self.features = self.feature_extractor["model"](image)

    def publish_segmentation(self):
        probabilities = self.sem_seg
        img = probabilities.get()
        img = np.transpose(img, (1, 2, 0)).astype(np.float32)
        seg_msg = self.cv_bridge.cv2_to_imgmsg(img, encoding="passthrough")
        seg_msg.header.frame_id = self.header.frame_id
        seg_msg.header.stamp = self.header.stamp
        self.seg_pub.publish(seg_msg)

    def publish_feature(self):
        features = self.features
        img = features.cpu().detach().numpy()
        img = np.transpose(img, (1, 2, 0)).astype(np.float32)
        feature_msg = self.cv_bridge.cv2_to_imgmsg(img, encoding="passthrough")
        feature_msg.header.frame_id = self.header.frame_id
        feature_msg.header.stamp = self.header.stamp
        self.feature_pub.publish(feature_msg)

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

    def publish_feature_image(self, features):
        data = np.reshape(features.cpu().detach().numpy(), (features.shape[0], -1)).T
        n_components = 3
        pca = PCA(n_components=n_components).fit(data)
        pca_descriptors = pca.transform(data)
        img_pca = pca_descriptors.reshape(features.shape[1], features.shape[2], n_components)
        comp = img_pca  # [:, :, -3:]
        comp_min = comp.min(axis=(0, 1))
        comp_max = comp.max(axis=(0, 1))
        comp_img = (comp - comp_min) / (comp_max - comp_min)
        comp_img = (comp_img * 255).astype(np.uint8)
        feat_msg = self.cv_bridge.cv2_to_imgmsg(comp_img, encoding="passthrough")
        feat_msg.header.frame_id = self.header.frame_id
        self.feat_im_pub.publish(feat_msg)


if __name__ == "__main__":
    arg = sys.argv[1]
    sensor_name = arg
    rospy.init_node("semantic_segmentation_node", anonymous=True, log_level=rospy.INFO)
    node = SemanticSegmentationNode(sensor_name)
    rospy.spin()
