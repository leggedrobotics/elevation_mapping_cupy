import rospy
import sys

import numpy as np
import cupy as cp

np.float = np.float64  # temp fix for following import suggested at https://github.com/eric-wieser/ros_numpy/issues/37
import ros_numpy
import matplotlib.pyplot as plt
from skimage.io import imshow

import message_filters
from sensor_msgs.msg import Image, CameraInfo
from sensor_msgs.msg import PointCloud2, Image
from cv_bridge import CvBridge

from semantic_sensor.pointcloud_parameters import PointcloudParameter
from semantic_sensor.networks import resolve_model
from semantic_sensor.utils import decode_max
from sklearn.decomposition import PCA


class PointcloudNode:
    def __init__(self, sensor_name):
        """Get parameter from server, initialize variables and semantics, register publishers and subscribers.

        Args:
            sensor_name (str): Name of the sensor in the ros param server.
        """
        # TODO: if this is going to be loaded from another package we might need to change namespace
        self.param: PointcloudParameter = PointcloudParameter()
        self.param.feature_config.input_size = [80, 160]
        namesp = rospy.get_name()
        if rospy.has_param(namesp + "/subscribers"):
            config = rospy.get_param(namesp + "/subscribers")
            self.param: PointcloudParameter = PointcloudParameter.from_dict(config[sensor_name])
        else:
            print("NO ROS ENV found.")

        self.param.sensor_name = sensor_name
        print("--------------Pointcloud Parameters-------------------")
        print(self.param.dumps_yaml())
        print("--------------End of Parameters-----------------------")
        self.semseg_color_map = None
        # setup custom dtype
        self.create_custom_dtype()
        # setup semantics
        self.feature_extractor = None
        self.semantic_model = None
        self.segmentation_channels = None
        self.feature_channels = None
        self.initialize_semantics()

        # setup pointcloud creation
        self.cv_bridge = CvBridge()
        self.P = None
        self.header = None
        self.register_sub_pub()
        self.prediction_img = None
        self.feat_img = None

    def initialize_semantics(self):
        """Resolve the feature and segmentation mode and create segmentation_channel and feature_channels.

        - segmentation_channels: is a dictionary that contains the segmentation channel names as key and the fusion algorithm as value.
        - feature_channels: is a dictionary that contains the features channel names as key and the fusion algorithm as value.
        """
        if self.param.semantic_segmentation:
            self.semantic_model = resolve_model(self.param.segmentation_model, self.param)
            self.segmentation_channels = {}
            for i, (chan, fusion) in enumerate(zip(self.param.channels, self.param.fusion)):
                if fusion in ["class_bayesian", "class_average", "class_max"]:
                    self.segmentation_channels[chan] = fusion
            assert len(self.segmentation_channels.keys()) > 0
        if self.param.feature_extractor:
            self.feature_extractor = resolve_model(self.param.feature_config.name, self.param.feature_config)
            self.feature_channels = {}
            for i, (chan, fusion) in enumerate(zip(self.param.channels, self.param.fusion)):
                if fusion in ["average"]:
                    self.feature_channels[chan] = fusion
            assert len(self.feature_channels.keys()) > 0

    def register_sub_pub(self):
        """Register publishers and subscribers."""
        # subscribers
        rospy.Subscriber(self.param.cam_info_topic, CameraInfo, self.cam_info_callback)
        rgb_sub = message_filters.Subscriber(self.param.image_topic, Image)
        depth_sub = message_filters.Subscriber(self.param.depth_topic, Image)
        if self.param.confidence:
            confidence_sub = message_filters.Subscriber(self.param.confidence_topic, Image)
            ts = message_filters.ApproximateTimeSynchronizer(
                [depth_sub, rgb_sub, confidence_sub,], queue_size=10, slop=0.5,
            )
        else:
            ts = message_filters.ApproximateTimeSynchronizer([depth_sub, rgb_sub,], queue_size=10, slop=0.5,)
        ts.registerCallback(self.image_callback)

        self.pcl_pub = rospy.Publisher(self.param.topic_name, PointCloud2, queue_size=2)
        # publishers
        if self.param.semantic_segmentation:
            if self.param.publish_segmentation_image:
                self.seg_pub = rospy.Publisher(self.param.segmentation_image_topic, Image, queue_size=2)
            if "class_max" in self.param.fusion:
                self.labels = self.semantic_model["model"].get_classes()
            else:
                self.labels = list(self.segmentation_channels.keys())
            self.semseg_color_map = self.color_map(len(self.labels))
            if self.param.show_label_legend:
                self.color_map_viz()
        if self.param.feature_extractor:
            # todo
            if True:
                self.feat_pub = rospy.Publisher(self.param.feature_config.feature_image_topic, Image, queue_size=2)

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
        cmap[1] = np.array([188, 63, 59])
        cmap[2] = np.array([81, 113, 162])
        cmap[3] = np.array([136, 49, 132])
        cmap = cmap / 255 if normalized else cmap
        return cmap[1:]

    def color_map_viz(self):
        """Display the color map of the classes."""
        nclasses = len(self.labels)
        row_size = 50
        col_size = 500
        if self.param.semantic_segmentation:
            cmap = self.semseg_color_map
        array = np.empty((row_size * (nclasses), col_size, cmap.shape[1]), dtype=cmap.dtype)
        for i in range(nclasses):
            array[i * row_size : i * row_size + row_size, :] = cmap[i]
        imshow(array)
        plt.yticks([row_size * i + row_size / 2 for i in range(nclasses)], self.labels)
        plt.xticks([])
        plt.show()

    def create_custom_dtype(self):
        """Generate a new dtype according to the channels in the params.

        Some channels might remain empty.
        """
        self.dtype = [
            ("x", np.float32),
            ("y", np.float32),
            ("z", np.float32),
        ]
        for chan, fus in zip(self.param.channels, self.param.fusion):
            self.dtype.append((chan, np.float32))
        print(self.dtype)

    def cam_info_callback(self, msg):
        """Subscribe to the camera infos to get projection matrix and header.

        Args:
            msg:
        """
        a = cp.asarray(msg.P)
        self.P = cp.resize(a, (3, 4))
        self.height = msg.height
        self.width = msg.width
        self.header = msg.header

    def image_callback(self, depth_msg, rgb_msg=None, confidence_msg=None):
        confidence = None
        image = None
        if self.P is None:
            return
        if rgb_msg is not None:
            image = cp.asarray(self.cv_bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8"))
        depth = cp.asarray(self.cv_bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough"))
        if confidence_msg is not None:
            confidence = cp.asarray(self.cv_bridge.imgmsg_to_cv2(confidence_msg, desired_encoding="passthrough"))

        pcl = self.create_pcl_from_image(image, depth, confidence)
        self.publish_pointcloud(pcl, depth_msg.header)

        if self.param.publish_segmentation_image:
            self.publish_segmentation_image(self.prediction_img)
            # todo
        if self.param.publish_feature_image and self.param.feature_extractor:
            self.publish_feature_image(self.feat_img)

    def create_pcl_from_image(self, image, depth, confidence):
        """Generate the pointcloud from the depth map and process the image.

        Args:
            image:
            depth:
            confidence:

        Returns:

        """
        u, v = self.get_coordinates(depth, confidence)

        # create pointcloud
        world_x = (u.astype(np.float32) - self.P[0, 2]) * depth[v, u] / self.P[0, 0]
        world_y = (v.astype(np.float32) - self.P[1, 2]) * depth[v, u] / self.P[1, 1]
        world_z = depth[v, u]
        points = np.zeros(world_x.shape, dtype=self.dtype)
        points["x"] = cp.asnumpy(world_x)
        points["y"] = cp.asnumpy(world_y)
        points["z"] = cp.asnumpy(world_z)
        self.process_image(image, u, v, points)
        return points

    def get_coordinates(self, depth, confidence):
        """Define which pixels are valid to generate the pointcloud.

        Args:
            depth:
            confidence:

        Returns:

        """
        pos = cp.where(depth > 0, 1, 0)
        low = cp.where(depth < 8, 1, 0)
        if confidence is not None:
            conf = cp.where(confidence >= self.param.confidence_threshold, 1, 0)
        else:
            conf = cp.ones(pos.shape)
        fin = cp.isfinite(depth)
        temp = cp.maximum(cp.rint(fin + pos + conf + low - 2.6), 0)
        mask = cp.nonzero(temp)
        u = mask[1]
        v = mask[0]
        return u, v

    def process_image(self, image, u, v, points):
        """Depending on setting generate color, semantic segmentation or feature channels.

        Args:
            image:
            u:
            v:
            points:
        """
        if "color" in self.param.fusion:
            valid_rgb = image[v, u].get()
            r = np.asarray(valid_rgb[:, 0], dtype=np.uint32)
            g = np.asarray(valid_rgb[:, 1], dtype=np.uint32)
            b = np.asarray(valid_rgb[:, 2], dtype=np.uint32)
            rgb_arr = np.array((r << 16) | (g << 8) | (b << 0), dtype=np.uint32)
            rgb_arr.dtype = np.float32
            position = self.param.fusion.index("color")
            points[self.param.channels[position]] = rgb_arr

        if self.segmentation_channels is not None:
            self.perform_segmentation(image, points, u, v)
        if self.feature_channels is not None:
            self.extract_features(image, points, u, v)

    def perform_segmentation(self, image, points, u, v):
        """Feedforward image through semseg NN and then append pixels to pcl and save image for publication.

        Args:
            image:
            points:
            u:
            v:
        """
        prediction = self.semantic_model["model"](image)
        values = prediction[:, v.get(), u.get()].get()
        for it, channel in enumerate(self.semantic_model["model"].actual_channels):
            points[channel] = values[it]
        if self.param.publish_segmentation_image and self.param.semantic_segmentation:
            self.prediction_img = prediction

    def extract_features(self, image, points, u, v):
        """Feedforward image through feature extraction NN and then append pixels to pcl.

        Args:
            image:
            points:
            u:
            v:
        """
        prediction = self.feature_extractor["model"](image)
        values = prediction[:, v.get(), u.get()].cpu().detach().numpy()
        for it, channel in enumerate(self.feature_channels.keys()):
            points[channel] = values[it]
            # todo
        if False and self.param.feature_extractor:
            self.feat_img = prediction

    def publish_segmentation_image(self, probabilities):
        if self.param.semantic_segmentation:
            colors = cp.asarray(self.semseg_color_map)
            assert colors.ndim == 2 and colors.shape[1] == 3
        if self.P is None:
            return
        prob = cp.zeros((len(self.labels),) + probabilities.shape[1:])
        if "class_max" in self.param.fusion:
            # decode, create an array with all possible classes and insert probabilities
            it = 0
            for iit, (chan, fuse) in enumerate(zip(self.param.channels, self.param.fusion)):
                if fuse in ["class_max"]:
                    temp = probabilities[it]
                    temp_p, temp_i = decode_max(temp)
                    temp_i.choose(prob)
                    c = cp.mgrid[0 : temp_i.shape[0], 0 : temp_i.shape[1]]
                    prob[temp_i, c[0], c[1]] = temp_p
                    it += 1
                elif fuse in ["class_bayesian", "class_average"]:
                    # assign fixed probability to correct index
                    if chan in self.semantic_model["model"].segmentation_channels:
                        prob[self.semantic_model["model"].segmentation_channels[chan]] = probabilities[it]
                        it += 1
            img = cp.argmax(prob, axis=0)

        else:
            img = cp.argmax(probabilities, axis=0)
        img = colors[img].astype(cp.uint8)  # N x H x W x 3
        img = img.get()
        seg_msg = self.cv_bridge.cv2_to_imgmsg(img, encoding="rgb8")
        seg_msg.header.frame_id = self.header.frame_id
        self.seg_pub.publish(seg_msg)

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
        self.feat_pub.publish(feat_msg)

    def publish_pointcloud(self, pcl, header):
        pc2 = ros_numpy.msgify(PointCloud2, pcl)
        pc2.header = header
        # pc2.header.frame_id = self.param.cam_frame
        self.pcl_pub.publish(pc2)

def main():
    arg = sys.argv[1]
    sensor_name = sys.argv[1]
    rospy.init_node("semantic_pointcloud_node", anonymous=True, log_level=rospy.INFO)
    node = PointcloudNode(sensor_name)
    rospy.spin()

if __name__ == "__main__":
    main()