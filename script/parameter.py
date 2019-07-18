import yaml
import numpy as np


class Parameter(object):
    def __init__(self):
        self.resolution = 0.02
        self.gather_mode = 'mean'

        self.map_length = 10.0
        self.sensor_noise_factor = 0.05
        self.mahalanobis_thresh = 2.0
        self.outlier_variance = 0.01
        self.time_variance = 0.01

        self.max_variance = 1.0
        self.dilation_size = 2

        self.traversability_inlier = 0.1
        self.wall_num_thresh = 100
        self.min_height_drift_cnt = 100

        self.enable_edge_sharpen = True

        self.initial_variance = 10.0
        self.w1 = np.zeros((4, 1, 3, 3))
        self.w2 = np.zeros((4, 1, 3, 3))
        self.w3 = np.zeros((4, 1, 3, 3))
        self.w_out = np.zeros((1, 12, 1, 1))

    def load_weights(self, filename):
        with open(filename) as file:
            weights = yaml.load(file)
            self.w1 = np.array(weights['w1'])
            self.w2 = np.array(weights['w2'])
            self.w3 = np.array(weights['w3'])
            self.w_out = np.array(weights['w_out'])

    def set_use_cupy(self, use_cupy):
        self.use_cupy = use_cupy

    def set_resolution(self, resolution):
        self.resolution = resolution

    def set_gather_mode(self, gather_mode):
        self.gather_mode = gather_mode

    def set_map_length(self, map_length):
        self.map_length = map_length

    def set_sensor_noise_factor(self, sensor_noise_factor):
        self.sensor_noise_factor = sensor_noise_factor

    def set_mahalanobis_thresh(self, mahalanobis_thresh):
        self.mahalanobis_thresh = mahalanobis_thresh

    def set_outlier_variance(self, outlier_variance):
        self.outlier_variance = outlier_variance

    def set_time_variance(self, time_variance):
        self.time_variance = time_variance

    def set_max_variance(self, max_variance):
        self.max_variance = max_variance

    def set_initial_variance(self, initial_variance):
        self.initial_variance = initial_variance

    def set_dilation_size(self, dilation_size):
        self.dilation_size = dilation_size

    def set_traversability_inlier(self, traversability_inlier):
        self.traversability_inlier = traversability_inlier

    def set_wall_num_thresh(self, wall_num_thresh):
        self.wall_num_thresh = wall_num_thresh

    def set_min_height_drift_cnt(self, min_height_drift_cnt):
        self.min_height_drift_cnt = min_height_drift_cnt

    def set_enable_edge_sharpen(self, enable_edge_sharpen):
        self.enable_edge_sharpen = enable_edge_sharpen
