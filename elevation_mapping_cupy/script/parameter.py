from dataclasses import dataclass
import pickle
import numpy as np

@dataclass
class Parameter:
    resolution: float = 0.02
    gather_mode: str = 'mean'

    map_length:float = 10.0
    sensor_noise_factor:float = 0.05
    mahalanobis_thresh:float = 2.0
    outlier_variance:float = 0.01
    drift_compensation_variance_inlier:float = 0.1
    time_variance:float = 0.01
    time_interval:float = 0.1

    max_variance:float = 1.0
    dilation_size:float = 2
    dilation_size_initialize:float = 10
    drift_compensation_alpha:float = 1.0

    traversability_inlier:float = 0.1
    wall_num_thresh:float = 100
    min_height_drift_cnt:float = 100

    max_ray_length:float = 2.0
    cleanup_step:float = 0.01
    cleanup_cos_thresh:float = 0.5
    min_valid_distance:float = 0.3
    max_height_range:float = 1.0
    ramped_height_range_a:float = 0.3
    ramped_height_range_b:float = 1.0
    ramped_height_range_c:float = 0.2

    safe_thresh:float = 0.5
    safe_min_thresh:float = 0.5
    max_unsafe_n:int = 20

    min_filter_size:int = 5
    min_filter_iteration:int = 3

    max_drift:float = 0.10

    overlap_clear_range_xy:float = 4.0
    overlap_clear_range_z:float = 2.0

    enable_edge_sharpen:bool = True
    enable_drift_compensation:bool = True
    enable_visibility_cleanup:bool = True
    enable_overlap_clearance:bool = True
    position_noise_thresh:float = 0.1
    orientation_noise_thresh:float = 0.1

    initial_variance:float = 10.0
    initialized_variance:float = 10.0
    w1:np.ndarray = np.zeros((4, 1, 3, 3))
    w2:np.ndarray = np.zeros((4, 1, 3, 3))
    w3:np.ndarray = np.zeros((4, 1, 3, 3))
    w_out:np.ndarray = np.zeros((1, 12, 1, 1))

    def load_weights(self, filename):
        with open(filename,'rb') as file:
            weights = pickle.load(file)
            self.w1 = weights['conv1.weight']
            self.w2 = weights['conv2.weight']
            self.w3 = weights['conv3.weight']
            self.w_out = weights['conv_final.weight']

    def get_names(self):
        return list(self.__annotations__.keys())

    def get_types(self):
        return [v.__name__ for v in self.__annotations__.values()]

    def set_value(self, name, value):
        setattr(self, name, value)

    # def set_use_cupy(self, use_cupy):
    #     self.use_cupy = use_cupy
    # 
    # def set_resolution(self, resolution):
    #     self.resolution = resolution
    # 
    # def set_gather_mode(self, gather_mode):
    #     self.gather_mode = gather_mode
    # 
    # def set_map_length(self, map_length):
    #     self.map_length = map_length
    # 
    # def set_sensor_noise_factor(self, sensor_noise_factor):
    #     self.sensor_noise_factor = sensor_noise_factor
    # 
    # def set_mahalanobis_thresh(self, mahalanobis_thresh):
    #     self.mahalanobis_thresh = mahalanobis_thresh
    # 
    # def set_outlier_variance(self, outlier_variance):
    #     self.outlier_variance = outlier_variance
    # 
    # def set_drift_compensation_variance_inlier(self, drift_compensation_variance_inlier):
    #     self.drift_compensation_variance_inlier = drift_compensation_variance_inlier
    # 
    # def set_time_variance(self, time_variance):
    #     self.time_variance = time_variance
    # 
    # def set_max_variance(self, max_variance):
    #     self.max_variance = max_variance
    # 
    # def set_initial_variance(self, initial_variance):
    #     self.initial_variance = initial_variance
    # 
    # def set_initialezed_variance(self, initialized_variance):
    #     self.initialized_variance = initialized_variance
    # 
    # def set_dilation_size(self, dilation_size):
    #     self.dilation_size = dilation_size
    # 
    # def set_dilation_size_initialize(self, dilation_size_initialize):
    #     self.dilation_size_initialize = dilation_size_initialize
    # 
    # def set_traversability_inlier(self, traversability_inlier):
    #     self.traversability_inlier = traversability_inlier
    # 
    # def set_position_noise_thresh(self, position_noise_thresh):
    #     self.position_noise_thresh = position_noise_thresh
    # 
    # def set_orientation_noise_thresh(self, orientation_noise_thresh):
    #     self.orientation_noise_thresh = orientation_noise_thresh
    # 
    # def set_wall_num_thresh(self, wall_num_thresh):
    #     self.wall_num_thresh = wall_num_thresh
    # 
    # def set_min_height_drift_cnt(self, min_height_drift_cnt):
    #     self.min_height_drift_cnt = min_height_drift_cnt
    # 
    # def set_max_ray_length(self, max_ray_length):
    #     self.max_ray_length = max_ray_length
    # 
    # def set_min_valid_distance(self, min_valid_distance):
    #     self.min_valid_distance = min_valid_distance
    # 
    # def set_max_height_range(self, max_height_range):
    #     self.max_height_range = max_height_range
    # 
    # def set_ramped_height_range_a(self, x):
    #     self.ramped_height_range_a = x
    # 
    # def set_ramped_height_range_b(self, x):
    #     self.ramped_height_range_b = x
    # 
    # def set_ramped_height_range_c(self, x):
    #     self.ramped_height_range_c = x
    # 
    # def set_cleanup_step(self, cleanup_step):
    #     self.cleanup_step = cleanup_step
    # 
    # def set_enable_edge_sharpen(self, enable_edge_sharpen):
    #     self.enable_edge_sharpen = enable_edge_sharpen
    # 
    # def set_enable_drift_compensation(self, enable_drift_compensation):
    #     self.enable_drift_compensation = enable_drift_compensation
    # 
    # def set_enable_visibility_cleanup(self, enable_visibility_cleanup):
    #     self.enable_visibility_cleanup = enable_visibility_cleanup
    # 
    # def set_safe_thresh(self, safe_thresh):
    #     self.safe_thresh = safe_thresh
    # 
    # def set_safe_min_thresh(self, safe_min_thresh):
    #     self.safe_min_thresh = safe_min_thresh
    # 
    # def set_max_unsafe_n(self, max_unsafe_n):
    #     self.max_unsafe_n = max_unsafe_n
    # 
    # def set_min_filter_size(self, x):
    #     self.min_filter_size = x
    # 
    # def set_min_filter_iteration(self, x):
    #     self.min_filter_iteration = x
    # 
    # def set_max_drift(self, x):
    #     self.max_drift = x
    # 
    # def set_drift_compensation_alpha(self, x):
    #     self.drift_compensation_alpha = x
    # 
    # def set_cleanup_cos_thresh(self, x):
    #     self.cleanup_cos_thresh = x
    # 
    # def set_time_interval(self, x):
    #     self.time_interval = x
    # 
    # def set_enable_overlap_clearance(self, x):
    #     self.enable_overlap_clearance = x
    # 
    # def set_overlap_clear_range_xy(self, x):
    #     self.overlap_clear_range_xy = x
    # 
    # def set_overlap_clear_range_z(self, x):
    #     self.overlap_clear_range_z = x


if __name__ == "__main__":
    param = Parameter()
    print(param)
    print(param.resolution)
    param.set_value('resolution', 0.1)
    print(param.resolution)

    print('names ', param.get_names())
    print('types ', param.get_types())
