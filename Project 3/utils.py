import numpy as np
import matplotlib.pyplot as plt
from transforms3d.euler import mat2euler


def load_data(file_name):
    """
  function to read visual features, IMU measurements and calibration parameters
  Input:
      file_name: the input data file. Should look like "XXX_sync_KLT.npz"
  Output:
      t: time stamp
          with shape 1*t
      features: visual feature point coordinates in stereo images,
          with shape 4*n*t, where n is number of features
      linear_velocity: IMU measurements in IMU frame
          with shape 3*t
      rotational_velocity: IMU measurements in IMU frame
          with shape 3*t
      K: (left)camera intrinsic matrix
          [fx  0 cx
            0 fy cy
            0  0  1]
          with shape 3*3
      b: stereo camera baseline
          with shape 1
      cam_T_imu: extrinsic matrix from IMU to (left)camera, in SE(3).
          close to
          [ 0 -1  0 t1
            0  0 -1 t2
            1  0  0 t3
            0  0  0  1]
          with shape 4*4
  """
    with np.load(file_name) as data:
        t = data["time_stamps"]  # time_stamps
        features = data["features"]  # 4 x num_features : pixel coordinates of features
        linear_velocity = data["linear_velocity"]  # linear velocity measured in the body frame
        rotational_velocity = data["rotational_velocity"]  # rotational velocity measured in the body frame
        K = data["K"]  # intrindic calibration matrix
        b = data["b"]  # baseline
        cam_T_imu = data["cam_T_imu"]  # Transformation from imu to camera frame
    return t, features, linear_velocity, rotational_velocity, K, b, cam_T_imu


def visualize_trajectory_2d(pose, pose2, landmark, landmark2, show_ori=False):
    """
  function to visualize the trajectory in 2D
  Input:
      pose:   4*4*N matrix representing the camera pose,
              where N is the number of pose, and each
              4*4 matrix is in SE(3)
  """
    fig, ax = plt.subplots(figsize=(8, 8))
    n_pose = pose.shape[2]
    ax.plot(pose[0, 3, :], pose[1, 3, :], 'r-', label="Prediction only")
    ax.plot(pose2[0, 3, :], pose2[1, 3, :], 'g--', label="VI SLAM")
    ax.plot(landmark[0, :], landmark[1, :], 'bo', markersize=1, label="landmark")
    ax.plot(landmark2[0, :], landmark[1, :], 'ko', markersize=1, label="updated landmark")
    ax.scatter(pose[0, 3, 0], pose[1, 3, 0], marker='s', label="start")
    ax.scatter(pose[0, 3, -1], pose[1, 3, -1], marker='o', label="end")

    if show_ori:
        select_ori_index = list(range(0, n_pose, max(int(n_pose / 50), 1)))
        yaw_list = []
        for i in select_ori_index:
            _, _, yaw = mat2euler(pose[:3, :3, i])
            yaw_list.append(yaw)
        dx = np.cos(yaw_list)
        dy = np.sin(yaw_list)
        dx, dy = [dx, dy] / np.sqrt(dx ** 2 + dy ** 2)
        ax.quiver(pose[0, 3, select_ori_index], pose[1, 3, select_ori_index], dx, dy, color="b", units="xy", width=1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axis('equal')
    ax.grid(False)
    ax.legend()
    plt.show(block=True)
    return fig, ax


def cov_visualize(t, cov1, cov2):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(t, np.log(np.linalg.norm(cov1, axis=(0, 1))), label='Prediction only')
    ax.plot(t, np.log(np.linalg.norm(cov2, axis=(0, 1))), label='Visual Inertial SLAM')
    ax.legend()
    ax.set_xlabel('Time / s')
    ax.set_ylabel('Norm / dB')
    ax.set_title("Covariance difference")
    ax.grid()
    plt.show()
    

def hat(u):
    if len(u) == 3:
        return np.array([[0, -u[2], u[1]],
                         [u[2], 0, -u[0]],
                         [-u[1], u[0], 0]])
    if len(u) == 6:
        return np.block([[hat(u[3:]), u[:3].reshape(3, 1)],
                         [np.zeros((1, 4))]])
    else:
        raise Exception("Error")


def c_hat(u):
    if len(u) == 6:
        return np.block([[hat(u[3:]), hat(u[:3])],
                         [np.zeros((3, 3)), hat(u[3:])]])
    else:
        raise Exception("Error")


def operator(s):
    return np.block([[np.eye(3), -hat(s[:3])],
                     [np.zeros((1, 6))]])


def projection(q):
    if len(q) == 4:
        return q / q[2, :]
    else:
        raise Exception("Error")


def proj_derivative(q):
    if len(q) == 4:
        return 1 / q[2] * np.array([[1, 0, -q[0] / q[2], 0],
                                    [0, 1, -q[1] / q[2], 0],
                                    [0, 0, 0, 0],
                                    [0, 0, -q[3] / q[2], 1]])
    else:
        raise Exception("Error")


# covert feature to world frame
def world_T_optical(features, world_T_cam, M, b):
    d = features[0, :] - features[2, :]
    x = (features[0, :] - M[0, 2]) * b / d
    y = (features[1, :] - M[1, 2]) * (-M[2, 3]) / (M[1, 1] * d)
    z = -M[2, 3] / d
    return world_T_cam @ np.vstack((x, y, z, np.ones(x.shape)))


# for part(b)
def jacobian(M, cam_T_world, features_len, feature, idx):
    DD = np.vstack((np.eye(3, 3), np.zeros((1, 3))))
    H = np.zeros((4 * len(idx), 3 * features_len))
    for i in range(len(idx)):
        H[i*4: (i+1)*4, idx[i]*3 : (idx[i]+1)*3] = \
            M @ proj_derivative(cam_T_world @ feature[:, i]) @ cam_T_world @ DD
    return H


# for part(c)
def jacobian2(M, cam_T_imu, imu_mean, mj):
    H = np.zeros((4 * mj.shape[1], 6))
    for i in range(mj.shape[1]):
        H[i*4 : (i+1)*4, :] = M @ proj_derivative(cam_T_imu @ imu_mean @ mj[:, i].reshape(4, 1)) \
                              @ cam_T_imu @ operator(imu_mean @ mj[:, i].reshape(4, 1))
    return H

