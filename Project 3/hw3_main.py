from utils import *
from scipy import linalg

from starter_code.utils import *

if __name__ == '__main__':
    # load data
    filename = "./data/0027.npz"
    t, features, linear_velocity, rotational_velocity, K, b, cam_T_imu = load_data(filename)
    features_len = features.shape[1]
    time_len = features.shape[2]
    # Down sample
    step = np.arange(0, features_len, 10)
    features = features[:, step, :]
    features_len = features.shape[1]

    # Initialization
    # imu
    imu_mean = np.eye(4, 4)  # at single time stamp
    imu_mean_next = np.eye(4, 4)
    imu_cov = np.zeros((6, 6, time_len))  # for part(a), prediction only
    imu_cov[:, :, 0] = np.eye(6, 6)
    imu_cov_next = np.zeros((6, 6, time_len))  # for part(c), Visual Inertial SLAM
    imu_cov_next[:, :, 0] = np.eye(6, 6)

    # landmark
    V = 10
    landmark_mean = -1 * np.ones((4, features_len))  # 4*M matrix
    landmark_mean_next = -1 * np.ones((4, features_len))
    landmark_cov = V * np.eye(3 * features_len, 3 * features_len)  # 3M*3M matrix
    landmark_cov_next = V * np.eye(3 * features_len, 3 * features_len)

    # trajectory
    trajectory = np.zeros((4, 4, time_len))  # for part(a), prediction only
    trajectory[:, :, 0] = imu_mean
    trajectory2 = np.zeros((4, 4, time_len))  # for part(c), Visual Inertial SLAM
    trajectory2[:, :, 0] = imu_mean_next

    # M matrix
    M = np.block([[K[0:2, 0:3]],
                  [K[0:2, 0:3]]])
    M = np.concatenate((M, np.array([0, 0, -K[0, 0] * b, 0]).reshape(4, 1)), axis=1)
    # D matrix
    D = np.kron(np.eye(features_len), np.vstack((np.eye(3), np.zeros((1, 3)))))

    for i in range(1, time_len):
        # (a) IMU Localization via EKF Prediction
        tau = t[0, i] - t[0, i - 1]  # time step
        v = linear_velocity[:, i].reshape(3, 1)  # velocity
        w = rotational_velocity[:, i].reshape(3, 1)  # angular velocity
        u = np.vstack((v, w))

        # imu mean and cov
        exp = linalg.expm(-tau * c_hat(u))
        W = tau ** 2 * np.diag(np.random.normal(0, V, 6))
        imu_mean = linalg.expm(-tau * hat(u)) @ imu_mean
        imu_cov[:, :, i] = exp @ imu_cov[:, :, i - 1] @ exp.T

        # update trajectory
        trajectory[:, :, i] = linalg.inv(imu_mean)

        # (b) Landmark Mapping via EKF Update
        # convert coordinates
        cam_T_world = cam_T_imu @ imu_mean
        world_T_cam = np.linalg.inv(cam_T_world)

        # initialize update step
        idx_list = np.empty(0, dtype=int)
        feature_list = np.empty((4, 0), dtype=float)

        feature = features[:, :, i]  # all the features at single time stamp
        idx = np.array(np.where(feature[0, :] != -1))  # available features idx
        if idx.size != 0:
            feature2 = feature[:, idx[0, :]]  # available features
            feature2 = world_T_optical(feature2, world_T_cam, M, b)  # available features now in world frame
            for j in range(idx.size):
                idx_list = np.append(idx_list, idx[0, j])
                feature_list = np.hstack((feature_list, feature2[:, j].reshape(4, 1)))
                # if the landmark is first seen, initialize it, otherwise update the its position
                if np.array_equal(landmark_mean[:, idx[0, j]], [-1, -1, -1, -1]):
                    landmark_mean[:, idx[0, j]] = feature2[:, j]

        if len(idx_list) != 0:
            tij = landmark_mean[:, idx_list].reshape(4, -1)  # if observation i corresponds to landmark j at time t
            z = feature[:, idx_list]
            z_hat = M @ projection(cam_T_world @ tij)
            H = jacobian(M, cam_T_world, features_len, tij, idx_list)
            Kt = landmark_cov @ H.T @ np.linalg.inv(H @ landmark_cov @ H.T + np.eye(4 * len(idx_list)) * V)
            update_landmark_mean = (landmark_mean.flatten('F') +
                             D @ Kt @ (z - z_hat).flatten('F')).reshape(4, -1, order='F')
            landmark_cov = (np.eye(3 * features_len) - Kt @ H) @ landmark_cov


    for i in range(1, time_len):
        # update landmarks
        tau = t[0, i] - t[0, i - 1]  # time step
        v = linear_velocity[:, i].reshape(3, 1)  # velocity
        w = rotational_velocity[:, i].reshape(3, 1)  # angular velocity
        u = np.vstack((v, w))
        exp = linalg.expm(-tau * c_hat(u))

        # convert coordinates
        cam_T_world = cam_T_imu @ imu_mean_next
        world_T_cam = np.linalg.inv(cam_T_world)

        # initialize update step
        idx_list = np.empty(0, dtype=int)
        feature_list = np.empty((4, 0), dtype=float)

        feature = features[:, :, i]  # all the features at single time stamp
        idx = np.array(np.where(feature[0, :] != -1))  # available features idx
        if idx.size != 0:
            feature2 = feature[:, idx[0, :]]  # available features
            feature2 = world_T_optical(feature2, world_T_cam, M, b)  # available features now in world frame
            for j in range(idx.size):
                idx_list = np.append(idx_list, idx[0, j])
                feature_list = np.hstack((feature_list, feature2[:, j].reshape(4, 1)))
                # if the landmark is first seen, initialize it, otherwise update the its position
                if np.array_equal(landmark_mean_next[:, idx[0, j]], [-1, -1, -1, -1]):
                    landmark_mean_next[:, idx[0, j]] = feature2[:, j]

        if len(idx_list) != 0:
            tij = landmark_mean_next[:, idx_list].reshape(4, -1)  # if observation i corresponds to landmark j at time t
            z = feature[:, idx_list]
            z_hat = M @ projection(cam_T_world @ tij)
            H = jacobian(M, cam_T_world, features_len, tij, idx_list)
            Kt = landmark_cov_next @ H.T @ np.linalg.inv(H @ landmark_cov_next @ H.T + np.eye(4 * len(idx_list)) * V)
            landmark_mean_next = (landmark_mean_next.flatten('F') +
                             D @ Kt @ (z - z_hat).flatten('F')).reshape(4, -1, order='F')
            landmark_cov_next = (np.eye(3 * features_len) - Kt @ H) @ landmark_cov_next

        # (c) Visual-Inertial SLAM
        # imu mean and cov
        imu_mean_next = linalg.expm(-tau * hat(u)) @ imu_mean_next
        imu_cov_next[:, :, i] = exp @ imu_cov_next[:, :, i - 1] @ exp.T

        tij = landmark_mean_next[:, idx_list].reshape(4, -1)  # use update landmarks
        z_next_hat = M @ projection(cam_T_imu @ imu_mean_next @ tij)
        z_next = feature[:, idx_list] + np.random.normal(0, V, z_next_hat.shape)
        H_next = jacobian2(M, cam_T_imu, imu_mean_next, tij)
        Kt_next = imu_cov_next[:, :, i] @ H_next.T @ np.linalg.inv(
            H_next @ imu_cov_next[:, :, i] @ H_next.T + np.eye(4 * len(idx_list)) * V)  # 6*4Nt
        imu_mean_next = linalg.expm(hat(Kt_next @ (z_next - z_next_hat).flatten('F'))) @ imu_mean_next  # 4*4
        imu_cov_next[:, :, i] = (np.eye(6) - Kt_next @ H_next) @ imu_cov_next[:, :, i]  # 6*6
        trajectory2[:, :, i] = np.linalg.inv(imu_mean_next)


    # Visual Inertial result
    visualize_trajectory_2d(trajectory, np.zeros(trajectory2.shape), landmark_mean, landmark_mean, show_ori=True)
    visualize_trajectory_2d(trajectory, trajectory2, landmark_mean, landmark_mean_next, show_ori=True)
    cov_visualize(t.squeeze(), imu_cov, imu_cov_next)
