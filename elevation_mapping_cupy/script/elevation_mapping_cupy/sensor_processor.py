#
# Copyright (c) 2024, W. Jacob Wagner. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import numpy as np
import cupy as cp
from cupyx.profiler import time_range


def make_3x1vec(vec):
    """Make sure vector is a 3x1 vector"""
    if vec.shape == (3,):
        vec = vec[:, np.newaxis]
    assert vec.shape == (3, 1), "Vector should be a 3x1 vector"
    return vec
class SensorProcessor(object):
    """
    Class for processing sensor data.
    Currenlty restricted to point cloud data.
    Mainly used for sensor dependent propagation of uncertainty.
    """
    @time_range('init', color_id=0)
    def __init__(self, sensor_ID, noise_model_name, noise_model_params, xp=cp, data_type=cp.float32):
        """Initialize sensor processor for a specific sensor.

        Args:
            sensor_ID (str): Sensor ID. Should be unique for each instance of a sensor
            noise_model_name (str):                     Noise model of the sensor. Either "SLS" or "LiDAR".
            noise_model_params (dict):                  Parameters for the noise model.
            xp (module):                                Numpy or Cupy module.
        """
        self.data_type = data_type
        self.sensor_ID = sensor_ID
        # Set default values for C_BS and B_r_BS
        self.C_BS = xp.eye(3, dtype=self.data_type)
        self.B_r_BS = make_3x1vec(xp.array([0.0, 0.0, 0.0],dtype=self.data_type))
        # Set default values for Sigma_Theta_BS and Sigma_b_r_BS
        # Sigma_Theta_BS: Covariance of sensor orientation where orientation is defined 
        #                 as fixed-axis(extrinsic) xyz(roll, pitch, yaw) Euler angles.
        self.Sigma_Theta_BS = xp.zeros((3, 3), dtype=self.data_type)
        # Sigma_b_r_BS:   Covariance of sensor position in base frame.
        self.Sigma_b_r_BS = xp.zeros((3, 3), dtype=self.data_type)

        self.noise_models = {"SLS": self.SLS_noise_model, # Structured Light Sensor Kinect v2 Fankhauser et al. 2015
                             "SLS_old": self.SLS_old_noise_model, # Isotropic variance proportional to square of distance in z direction
                             "constant": self.constant_noise_model, # Constant variance for all points
                             "LiDAR": self.LiDAR_noise_model} # LiDAR
        assert noise_model_name in self.noise_models.keys(), "noise_model should be chosen from {}".format(self.noise_models.keys())
        self.noise_model_name = noise_model_name
        self.noise_model = self.noise_models[self.noise_model_name]
        self.noise_model_params = {}
        # Make sure params are compatible with cupy
        for key in noise_model_params.keys():
            if key == "Sigma_Theta_BS_diag":
                assert len(noise_model_params[key]) == 3, "Sigma_Theta_BS_diag should be 3 element arrays"
                self.noise_model_params['Sigma_Theta_BS'] = xp.diag(cp.asarray(noise_model_params[key], dtype=self.data_type))
            elif key == "Sigma_b_r_BS_diag":
                assert len(noise_model_params[key]) == 3, "Sigma_b_r_BS_diag should be 3 element arrays"
                self.noise_model_params['Sigma_b_r_BS'] = xp.diag(cp.asarray(noise_model_params[key], dtype=self.data_type))
            else:
                self.noise_model_params[key] = xp.array(noise_model_params[key], dtype=self.data_type)
        self.xp = xp
    
    @time_range('set_BS_transform', color_id=0)
    def set_BS_transform(self, C_BS, B_r_BS):
        """Set the transformation from body frame to sensor frame
        Enables the transformation of points from sensor frame to body frame using:
        B_r_BP = C_BS * S_r_SP + B_r_BS

        Args: 
            C_BS (cupy._core.core.ndarray):             Matrix representing relative orientation of sensor frame in body frame.
                                                        (takes points from sensor frame and transforms them to body frame)
            B_r_BS (cupy._core.core.ndarray):           Position of sensor in body frame."""
        assert C_BS.shape == (3, 3), "C_BS should be a 3x3 matrix"
        self.C_BS = cp.asarray(C_BS, dtype=self.data_type)
        self.B_r_BS = cp.asarray(make_3x1vec(B_r_BS), dtype=self.data_type)
    
    @time_range('SS', color_id=0)
    def SS(self, v):
        """Skew symmetric matrix of a vector v"""
        C = self.xp.zeros((3, 3), dtype=self.data_type)
        C[0, 1] = -v[2]
        C[0, 2] = v[1]
        C[1, 0] = v[2]
        C[1, 2] = -v[0]
        C[2, 0] = -v[1]
        C[2, 1] = v[0]
        return C
        # Implementing this way instead of below because of cupy compatibility
        # return self.xp.array([[0, -v[2], v[1]],
        #                 [v[2], 0, -v[0]],
        #                 [-v[1], v[0], 0]])
    
    @time_range('ei', color_id=0)
    def ei(self, i):
        """Basis vector ei"""
        return self.xp.array([1 if j == i else 0 for j in range(3)], dtype=self.data_type)
    
    @time_range('Rx', color_id=0)
    def Rx(self, a):
        """Rotation matrix around x-axis"""
        C = self.xp.eye(3, dtype=self.data_type)
        C[1:, 1:] = self.xp.array( [[self.xp.cos(a), -self.xp.sin(a)],
                                    [self.xp.sin(a), self.xp.cos(a)]],
                                    dtype=self.data_type)
        return C
        # Implementing this way instead of below because of cupy compatibility
        # return self.xp.array([[1, 0, 0],
        #                       [0, self.xp.cos(a), -self.xp.sin(a)],
        #                       [0, self.xp.sin(a), self.xp.cos(a)]])

    @time_range('Ry', color_id=0)
    def Ry(self, b):
        """Rotation matrix around y-axis"""
        C = self.xp.eye(3, dtype=self.data_type)
        C[0, 0] = self.xp.cos(b)
        C[0, 2] = self.xp.sin(b)
        C[2, 0] = -self.xp.sin(b)
        C[2, 2] = self.xp.cos(b)
        return C
        # Implementing this way instead of below because of cupy compatibility
        # return self.xp.array([[self.xp.cos(b), 0, self.xp.sin(b)],
        #                       [0, 1, 0],
        #                       [-self.xp.sin(b), 0, self.xp.cos(b)]])

    @time_range('Rz', color_id=0)
    def Rz(self, c):
        """Rotation matrix around z-axis"""
        C = self.xp.eye(3)
        C[:2, :2] = self.xp.array( [[self.xp.cos(c), -self.xp.sin(c)],
                                    [self.xp.sin(c), self.xp.cos(c)]],
                                    dtype=self.data_type)
        return C
        # Implementing this way instead of below because of cupy compatibility
        # return self.xp.array([[self.xp.cos(c), -self.xp.sin(c), 0],
        #                       [self.xp.sin(c), self.xp.cos(c), 0],
        #                       [0, 0, 1]])
    
    @time_range('diag_array_to_stacks', color_id=0)
    def diag_array_to_stacks(self, d):
        """Converts a 2D array where each row defines a diagonal into a 3D array of diagonal matrices

        Args:
            d: A 2D numpy array where each row defines a diagonal for a new matrix

        Returns:
            A 3D numpy array where D[:,:,i] is a diagonal matrix with d[i,:] on the diagonal
        """
        # TODO: Figure out how to do this without a for loop
        # Can't seem to pass dtype=self.data_type, but should be fine
        return self.xp.stack([self.xp.diag(val) for val in d], axis=0)
    
    @time_range('SLS_noise_model', color_id=0)
    def SLS_noise_model(self, S_r_SP, C_MB):
        """
        Structured Light Sensor noise model.
        Based on the paper:
        "Kinect v2 for mobile robot navigation: Evaluation and modeling" by
        P. Fankhauser, M. Bloesch, D. Rodriguez, R. Kaestner, M. Hutter, R. Siegwart,
        ICAR 2015.
        The axis convenciton assumes is z axis forward, x axis right, and y axis down.
        Using the model for sensor noise of the Kinect v2.
        sigma_l = a
        sigma_z = b - c * z + d * z^2(1 + e * cos(alpha)) + f * z^(3/2) * (theta^2)/(pi/2-theta)^2
        Assume theta and alpha are 0 this reduces to:
        sigma_z = b - c * z + 2 * d * z^2
        where z is the distance of the point from the sensor along the z axis of the sensor frame in meters,
        theta is the angle of the target with repsect to the z axis of the sensor frame (assumed to be 0),
        alpha is the angle of incidence of the light on the target (assumed to be 0),
        and a, b, c, d, e, and f are postive parameters that are determined experimentally.
        sigma_z and sigma_l are the standard deviations (in meters) of the noise in the z direction and the lateral direction.
        """
        sigma_l = self.noise_model_params["a"]
        z = S_r_SP[:,2]
        N = S_r_SP.shape[0]
        sigma_z = self.noise_model_params["b"] - self.noise_model_params["c"] * z + 2 * self.noise_model_params["d"] * z**2
        # Repeat sigma_l for each point
        sigma_l = self.xp.repeat(sigma_l, N)
        # Sigma_S_r_SP = self.xp.diag(self.xp.array([sigma_l**2, sigma_l**2, sigma_z**2]))
        Sigma_S_r_SP = self.diag_array_to_stacks(self.xp.array([sigma_l**2, sigma_l**2, sigma_z**2]).T)
        J_S_r_SP = C_MB @ self.C_BS

        return J_S_r_SP, Sigma_S_r_SP
    
    @time_range('SLS_old_noise_model', color_id=0)
    def SLS_old_noise_model(self, S_r_SP, C_MB):
        """
        Noise model assumed by em_cupy  prior to the addition of the SensorProcessor class.
        This method is kept for backwards compatibility. Assumes an isotropic variance 
        proportional to the square of the distance from the sensor along the z direction (front),
        of the camera. 
        """
        # Error propagation can be ignored for this approach as the variance is isotropic and
        # any rotations will not change the shape of the distribution
        J_S_r_SP = None
        z_noise = self.noise_model_params["sensor_noise_factor"] * self.xp.power(S_r_SP[:, 2], 2, dtype=self.data_type)
        N = S_r_SP.shape[0]
        Sigma_S_r_SP = self.diag_array_to_stacks(self.xp.array([z_noise, z_noise, z_noise]).T)
        return J_S_r_SP, Sigma_S_r_SP
    
    @time_range('constant_noise_model', color_id=0)
    def constant_noise_model(self, S_r_SP, C_MB):
        """
        Assume constant variance for all points
        """
        # Error propagation can be ignored for this approach as the variance is isotropic and
        # any rotations will not change the shape of the distribution
        J_S_r_SP = None
        var = self.noise_model_params["constant_variance"]
        N = S_r_SP.shape[0]
        Sigma = self.xp.diag(self.xp.array([var, var, var], dtype=self.data_type))
        Sigma_S_r_SP = self.xp.repeat(Sigma[np.newaxis, :, :], N, axis=0)
        return J_S_r_SP, Sigma_S_r_SP
    
    @time_range('LiDAR_noise_model', color_id=0)
    def LiDAR_noise_model(self, points, C_MB):
        raise NotImplementedError("LiDAR noise model not implemented yet")
    
    # TODO: May need to make this its own kernel...
    @time_range('get_ext_euler_angles', color_id=0)
    def get_ext_euler_angles(self, C):
        """
        Extract Extrinsically defined Euler angles from rotation matrix
        The transformation matrix C in terms of
        extrinsically defined Euler angles (roll, pitch, yaw)
        is defined as C = Rz(yaw) @ Ry(pitch) @ Rx(roll)
        i.e. first roll about x, then pitch about y, and then yaw about z (in that order)
        i.e. the rotation matrix is defined as a sequence of
        rotations about the axes of the original coordinate
        system where the axes are fixed in space.
        For reference Basic rotation matrices are defined as:
        Rx = lambda a: xp.array([[1, 0, 0],
                                [0, xp.cos(a), xp.sin(a)],
                                [0, -xp.sin(a), xp.cos(a)]])

        Ry = lambda b: xp.array([[xp.cos(b), 0, -xp.sin(b)],
                                [0, 1, 0],
                                [xp.sin(b), 0, xp.cos(b)]])

        Rz = lambda c: xp.array([[xp.cos(c), xp.sin(c), 0],
                                [-xp.sin(c), xp.cos(c), 0],
                                [0, 0, 1]])
        Note that this may be refered to as Extrinsic Tait-Bryan angles in x-y-z order.
        Using Reference: Computing Euler Angles from a Rotation Matrix by Gregory G. Slabaugh
        https://eecs.qmul.ac.uk/~gslabaugh/publications/euler.pdf

        2 solutions for these Extrinsic Tait-Bryan angles always exist. 
        The Eigen geometry module function eulerAngles(ax1,ax2,ax3) returns the intrinsically
        defined Euler angles (a,b,c) applied in the order ax1, ax2, ax3. and ensures that the angles
        (a,b,c) are in the ranges [0:pi]x[-pi:pi]x[-pi:pi].
        (https://eigen.tuxfamily.org/dox/group__Geometry__Module.html#title20)
        In order to obtain Extrinsic Tait-Bryan angles applied in the order x-y-z, robot_localization
        uses the Eigen function y,p,r = eulerAngles(2,1,0)
        (https://github.com/cra-ros-pkg/robot_localization/blob/49aab740c0b66c9f266141522e86d64dc86c8939/src/ros_robot_localization_listener.cpp#L456)
        which means that the orientation state being tracked by robot_localization
        (r,p,y) are in [-pi:pi]x[-pi:pi]x[0:pi].
        Given this assumption we can select the correct set of angles
        """
        if self.xp.abs(C[2, 0]) != 1.0:
            pitch1 = -self.xp.arcsin(C[2,0]) # Different compared to paper due to 
            pitch2 = self.xp.pi - pitch1
            roll1 = self.xp.arctan2(C[2,1]/self.xp.cos(pitch1), C[2,2]/self.xp.cos(pitch1))
            roll2 = self.xp.arctan2(C[2,1]/self.xp.cos(pitch2), C[2,2]/self.xp.cos(pitch2))
            yaw1 = self.xp.arctan2(C[1,0]/self.xp.cos(pitch1), C[0,0]/self.xp.cos(pitch1))
            yaw2 = self.xp.arctan2(C[1,0]/self.xp.cos(pitch2), C[0,0]/self.xp.cos(pitch2))
            # Select the correct set of angles
            a1_valid = roll1 >= -self.xp.pi and roll1 <= self.xp.pi
            a1_valid = a1_valid and (pitch1 >= -self.xp.pi and pitch1 <= self.xp.pi)
            a1_valid = a1_valid and (yaw1 >= 0 and yaw1 <= self.xp.pi)
            a2_valid = roll2 >= -self.xp.pi and roll2 <= self.xp.pi
            a2_valid = a2_valid and (pitch2 >= -self.xp.pi and pitch2 <= self.xp.pi)
            a2_valid = a2_valid and (yaw2 >= 0 and yaw2 <= self.xp.pi)
            if a1_valid:
                roll, pitch, yaw = roll1, pitch1, yaw1
            elif a2_valid:
                roll, pitch, yaw = roll2, pitch2, yaw2
            else:
                raise ValueError("No valid set of Euler angles found")
                
        else: # Gimbal lock: pitch is at -90 or 90 degrees
            yaw = 0.0 # This can be any value, but we choose 0.0 for consistency
            if C[2, 0] == -1.0:
                pitch = self.xp.pi/2
                roll = yaw + self.xp.arctan2(C[0,1], C[0,2])
            else:
                pitch = -self.xp.pi/2
                roll = -yaw + self.xp.arctan2(-C[0,1], -C[0,2])
        return roll, pitch, yaw

    @time_range('get_z_variance', color_id=0)
    def get_z_variance(self, points, C_MB, B_r_MB, Sigma_Theta_MB, Sigma_b_r_MB):
        """
        Calculate variance in z direction of map frame for each point.
        A sensor observes a point in the sensor frame.
        The point is then transformed to the map frame via:
        M_r_MP = C_MB @ (C_BS @ S_r_SP +  B_r_BS) + B_r_MB
        and projected into the z direction of the map frame via:
        z = H @ M_r_MP, H = [0, 0, 1]
        Uncertianty in the transform from the map frame to the base frame,
        the transform from the base frame to the sensor frame (mounting uncertainty),
        and the uncertainty of the measurement in the sensor frame are propagated
        through this equation to determine the uncertainty in the z direction of the map frame.
        This is done by computing the Jacobian of the transformation with respect to each of the
        variables and then using the covariance matrices to determine the variance in z.

        Args:
            points (cupy._core.core.ndarray):           Points in sensor frame. AKA S_r_SP
            C_MB (cupy._core.core.ndarray):             Matrix representing relative orientation of body frame in map frame.
            B_r_MB (cupy._core.core.ndarray):           Position of body frame in map frame.
            Sigma_Theta_MB (cupy._core.core.ndarray):   Covariance of base orientation in map frame where orientation is defined 
                                                        as fixed-axis(extrinsic) xyz(roll, pitch, yaw) Euler angles.
            Sigma_b_r_MB (cupy._core.core.ndarray):     Covariance of base position in map frame.

        Returns:
            cupy._core.core.ndarray: Variance of each point along z direction of map frame.
        """
        # C_MB, B_r_MB, Sigma_Theta_MB, Sigma_b_r_MB
        assert C_MB.shape == (3, 3), "C_MB should be a 3x3 matrix"
        C_MB = cp.asarray(C_MB, dtype=self.data_type)
        B_r_MB = cp.asarray(make_3x1vec(B_r_MB), dtype=self.data_type)
        assert Sigma_Theta_MB.shape == (3, 3), "Sigma_Theta_MB should be a 3x3 matrix"
        Sigma_Theta_MB = cp.asarray(Sigma_Theta_MB, dtype=self.data_type)
        assert Sigma_b_r_MB.shape == (3, 3), "Sigma_b_r_MB should be a 3x3 matrix"
        Sigma_b_r_MB = cp.asarray(Sigma_b_r_MB, dtype=self.data_type)
        assert points.shape[1] == 3, "Points should be a Nx3 matrix"
        S_r_SP = cp.asarray(points, dtype=self.data_type) # For ease of numpy/cupy notation points are Nx3 i.e. batch index first

        # If using the old SLS noise model, we can ignore error propagation
        if self.noise_model_name == "SLS_old":
            J_S_r_SP, Sigma_S_r_SP = self.noise_model(S_r_SP, C_MB)
            return Sigma_S_r_SP[:, 2, 2]
        elif self.noise_model_name == "constant":
            J_S_r_SP, Sigma_S_r_SP = self.noise_model(S_r_SP, C_MB)
            return Sigma_S_r_SP[:, 2, 2]
        
        # Error propagation
        # J_M_r_MB = I # Jacobian of M_r_MP wrt M_r_MB
        B_r_BP = (self.C_BS @ S_r_SP.T + self.B_r_BS).T # B_r_BP = C_BS @ S_r_SP +  B_r_BS
        J_Theta_MB = self.Jac_of_rot(C_MB, B_r_BP) # Jacobian of M_r_MP wrt Theta_MB
        J_B_r_BS = C_MB # Jacobian of M_r_MP wrt B_r_BS
        J_Theta_BS = C_MB @ self.Jac_of_rot(self.C_BS, S_r_SP) # Jacobian of M_r_MP wrt Theta_BS
        J_S_r_SP, Sigma_S_r_SP = self.noise_model(S_r_SP, C_MB) # Jacobian of M_r_MP wrt S_r_SP

        Sigma_Theta_BS = self.noise_model_params["Sigma_Theta_BS"]
        Sigma_b_r_BS = self.noise_model_params["Sigma_b_r_BS"]

        # Propagate uncertainty
        # Could apply H here, but in the future we may want the full
        # covariance matrix for applying a single point observation to multiple cells
        Sigma_M_r_MP =  Sigma_b_r_MB + \
                        J_Theta_MB @ Sigma_Theta_MB @ J_Theta_MB.transpose((0,2,1)) + \
                        J_B_r_BS @ Sigma_b_r_BS @ J_B_r_BS.T + \
                        J_Theta_BS @ Sigma_Theta_BS @ J_Theta_BS.transpose((0,2,1)) + \
                        J_S_r_SP @ Sigma_S_r_SP @ J_S_r_SP.T
        
        # Pull out the z variance
        z_var = Sigma_M_r_MP[:, 2, 2]
        return z_var

    @time_range('Jac_of_rot', color_id=0)
    def Jac_of_rot(self, C, r):
        """
        Calculate Jacobian of rotation matrix C applied to vector r
        with respect to extrinsic (fixed-axis) Euler angles (roll, pitch, yaw) x-y-z.
        The rotation matrix is represented as a 3x3 matrix.
        The vector is represented as a 3x1 vector.
        The Jacobian is a 3x3 matrix.
        Using method from "Differentiation of the Orientation Matrix by Matrix Multipliers" by J. Lucas, 1963
        https://www.asprs.org/wp-content/uploads/pers/1963journal/jul/1963_jul_708-715.pdf

        Args:
            C (cupy._core.core.ndarray): 3x3 rotation matrix.
            r (cupy._core.core.ndarray): nx3 vector.

        Returns:
            cupy._core.core.ndarray: nx3x3 Jacobian matrix. For ith point Row j of dCr[i] include the partial derivative]
                                     of the of [C*r]_j with respect to the roll, pitch, and yaw angles.
        """
        # First extract the roll extrinsic Euler angle from C
        # Ideally these would be provided as inputs, but we can extract them here
        roll, _, _ = self.get_ext_euler_angles(C)
        Qj = (self.Rx(roll).T) @ (self.SS(self.ei(1))) @ self.Rx(roll)
        dC_dyaw = self.SS(self.ei(2)) @ C
        dC_dpitch = C @ Qj
        dC_droll = C @ (self.SS(self.ei(0)))

        dCr_droll = (dC_droll @ r.T).T
        dCr_dpitch = (dC_dpitch @ r.T).T
        dCr_dyaw = (dC_dyaw @ r.T).T

        dCr = self.xp.stack([dCr_droll, dCr_dpitch, dCr_dyaw], axis=-1)
        return dCr # nx3x3 Jacobian matrix


if __name__ == "__main__":
    print("Starting")
    xp = cp
    sensor_ID = "test_sls"
    C_BS = xp.eye(3)*1.0e0
    B_r_BS = xp.array([0.0, 0.0, 0.0])
    Sigma_Theta_BS_diag = xp.array([1.0e-1, 1.0e-1, 1.0e-1])
    Sigma_b_r_BS_diag = xp.array([1.0e-3, 1.0e-3, 1.0e-3])
    noise_model_name = "SLS"
    # Taking parameters from equaiton 6 in the paper
    # sigma_z = b - c * z + 2 * d * z^2
    SLS_params = {"a": 6.8e-3,
                  "b": 28.0e-3,
                  "c": 38.0e-3,
                  "d": (42e-3+2.0e-3)/2.0,
                  "Sigma_Theta_BS_diag": Sigma_Theta_BS_diag,
                  "Sigma_b_r_BS_diag": Sigma_b_r_BS_diag}
    sp = SensorProcessor(sensor_ID, noise_model_name, SLS_params, xp)
    sp.set_BS_transform(C_BS, B_r_BS)
    # points = xp.array([[0, 0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0], [1.0, 2.0, 3.0]])
    # Generate a larger number of points randomly in range of -20 to 20 in each dimension
    points = xp.random.uniform(-20, 20, (1000, 3))
    C_MB = xp.eye(3)*1.0e0
    B_r_MB = xp.array([10.0, 0, 0])
    Sigma_Theta_MB = xp.eye(3)*1.0e-6
    Sigma_b_r_MB = xp.eye(3)*1.0e-1
    # TODO: Deal with data types
    for i in range(10):
        z_var = sp.get_z_variance(points, C_MB, B_r_MB, Sigma_Theta_MB, Sigma_b_r_MB)
    # print(z_var)
    # if xp == cp:
    #     from cupyx.profiler import benchmark
    #     print(benchmark(sp.get_z_variance, (points, C_MB, B_r_MB, Sigma_Theta_MB, Sigma_b_r_MB), n_repeat=20))
    print("Done!")