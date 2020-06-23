import pickle
import numpy as np


class Parameter(object):
    def __init__(self):
        self.resolution = 0.02
        self.gather_mode = 'mean'

        self.map_length = 10.0
        self.sensor_noise_factor = 0.05
        self.mahalanobis_thresh = 2.0
        self.outlier_variance = 0.01
        self.drift_compensation_variance_inlier = 0.1
        self.time_variance = 0.01
        self.time_interval = 0.1

        self.max_variance = 1.0
        self.dilation_size = 2
        self.dilation_size_initialize = 10

        self.traversability_inlier = 0.1
        self.wall_num_thresh = 100
        self.min_height_drift_cnt = 100

        self.max_ray_length = 2.0
        self.cleanup_step = 0.01
        self.cleanup_cos_thresh = 0.5
        self.min_valid_distance = 0.3
        self.max_height_range = 1.0

        self.safe_thresh = 0.5
        self.safe_min_thresh = 0.5
        self.max_unsafe_n = 20

        self.min_filter_size = 5
        self.min_filter_iteration = 3

        self.max_drift = 0.10

        self.enable_edge_sharpen = True
        self.enable_drift_compensation = True
        self.enable_visibility_cleanup = True
        self.position_noise_thresh = 0.1
        self.orientation_noise_thresh = 0.1

        self.initial_variance = 10.0
        self.initialized_variance = 10.0
        self.w1 = np.zeros((4, 1, 3, 3))
        self.w2 = np.zeros((4, 1, 3, 3))
        self.w3 = np.zeros((4, 1, 3, 3))
        self.w_out = np.zeros((1, 12, 1, 1))

    def load_weights(self, filename):
        with open(filename) as file:
            weights = pickle.load(file)
            self.w1 = weights['conv1.weight']
            self.w2 = weights['conv2.weight']
            self.w3 = weights['conv3.weight']
            self.w_out = weights['conv_final.weight']

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

    def set_drift_compensation_variance_inlier(self, drift_compensation_variance_inlier):
        self.drift_compensation_variance_inlier = drift_compensation_variance_inlier

    def set_time_variance(self, time_variance):
        self.time_variance = time_variance

    def set_max_variance(self, max_variance):
        self.max_variance = max_variance

    def set_initial_variance(self, initial_variance):
        self.initial_variance = initial_variance

    def set_initialezed_variance(self, initialized_variance):
        self.initialized_variance = initialized_variance

    def set_dilation_size(self, dilation_size):
        self.dilation_size = dilation_size

    def set_dilation_size_initialize(self, dilation_size_initialize):
        self.dilation_size_initialize = dilation_size_initialize

    def set_traversability_inlier(self, traversability_inlier):
        self.traversability_inlier = traversability_inlier

    def set_position_noise_thresh(self, position_noise_thresh):
        self.position_noise_thresh = position_noise_thresh

    def set_orientation_noise_thresh(self, orientation_noise_thresh):
        self.orientation_noise_thresh = orientation_noise_thresh

    def set_wall_num_thresh(self, wall_num_thresh):
        self.wall_num_thresh = wall_num_thresh

    def set_min_height_drift_cnt(self, min_height_drift_cnt):
        self.min_height_drift_cnt = min_height_drift_cnt

    def set_max_ray_length(self, max_ray_length):
        self.max_ray_length = max_ray_length

    def set_min_valid_distance(self, min_valid_distance):
        self.min_valid_distance = min_valid_distance

    def set_max_height_range(self, max_height_range):
        self.max_height_range = max_height_range

    def set_cleanup_step(self, cleanup_step):
        self.cleanup_step = cleanup_step

    def set_enable_edge_sharpen(self, enable_edge_sharpen):
        self.enable_edge_sharpen = enable_edge_sharpen

    def set_enable_drift_compensation(self, enable_drift_compensation):
        self.enable_drift_compensation = enable_drift_compensation

    def set_enable_visibility_cleanup(self, enable_visibility_cleanup):
        self.enable_visibility_cleanup = enable_visibility_cleanup

    def set_safe_thresh(self, safe_thresh):
        self.safe_thresh = safe_thresh

    def set_safe_min_thresh(self, safe_min_thresh):
        self.safe_min_thresh = safe_min_thresh

    def set_max_unsafe_n(self, max_unsafe_n):
        self.max_unsafe_n = max_unsafe_n

    def set_min_filter_size(self, x):
        self.min_filter_size = x

    def set_min_filter_iteration(self, x):
        self.min_filter_iteration = x

    def set_max_drift(self, x):
        self.max_drift = x

    def set_cleanup_cos_thresh(self, x):
        self.cleanup_cos_thresh = x

    def set_time_interval(self, x):
        self.time_interval = x
