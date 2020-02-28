import load_data as ld
import numpy as np
import matplotlib.pyplot as plt
import ParticleFilter as PF
import p2_utils as p2
import mapping
import os

### initialization
# Load data
joint = ld.get_joint("joint/train_joint0")
joint_angles = joint['head_angles']
joint_neck = joint_angles[0]  # z aex
joint_head = joint_angles[1]  # location aex
joint_time = joint['ts']
lidar = ld.get_lidar("lidar/train_lidar0")
lidar_pose = [x['delta_pose'] for x in lidar]
lidar_scan = [x['scan'] for x in lidar]
lidar_time = [x['t'] for x in lidar]
time_len = len(lidar_time)

# Synchronize timestamp
idx = []
for i in lidar_time:
    abs_value = abs(joint_time.squeeze() - i.squeeze())
    temp = np.where(abs_value == min(abs_value))
    idx.append(temp[0][0])

# New data with the same timestamp
joint_neck = joint_neck[idx]
joint_head = joint_head[idx]
joint_time = joint_time.squeeze()[idx]

# MAP
MAP = {}
MAP['res'] = 0.1  # meters
MAP['xmin'] = -30  # meters
MAP['ymin'] = -30
MAP['xmax'] = 30
MAP['ymax'] = 30
MAP['sizex'] = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1))  # numbers of cells
MAP['sizey'] = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
MAP['map'] = np.zeros((MAP['sizex'], MAP['sizey']))

# For map correlation
x_range = np.arange(-0.4, 0.5, 0.1)
y_range = np.arange(-0.4, 0.5, 0.1)
x_im = np.arange(MAP['xmin'], MAP['xmax'] + MAP['res'], MAP['res'])
y_im = np.arange(MAP['ymin'], MAP['ymax'] + MAP['res'], MAP['res'])


# Initialize particles and trajectory
N = 5  # number of particles
particle = np.zeros((3, N))  # particle position, where particle = (x, y, theta)'* N
weight = np.array([1 / N] * N).reshape(1, N)  # weight equally distribute at the beginning
trajectory = np.zeros((2, 1))

# Others
angles = np.arange(-135, 135, 270 / 1081) / 180 * np.pi
zero = np.array([0, 0, 0]).reshape(1, 3) # for creating matrix below
p = np.array([0, 0, 0.48, 1]).reshape(4, 1)
pp = np.array([0, 0, 0, 1]).reshape(4, 1)

# Downsample
step = 100
new_time = np.arange(0, time_len, step)

for i in new_time:
    # bTl
    yaw = np.array(
        [[np.cos(joint_neck[i]), -np.sin(joint_neck[i]), 0],
         [np.sin(joint_neck[i]), np.cos(joint_neck[i]), 0], [0, 0, 1]])
    pitch = np.array([[np.cos(joint_head[i]), 0, np.sin(joint_head[i])], [0, 1, 0],
                      [-np.sin(joint_head[i]), 0, np.cos(joint_head[i])]])
    R = np.dot(yaw, pitch)
    bTl = np.concatenate([np.concatenate([R, zero], axis=0), p], axis=1)  # 4*4, from lidar to  body

    # Find valid data
    ranges = np.double(lidar_scan[i].squeeze())
    indValid = np.logical_and((ranges < 30), (ranges > 0.1))  # take valid indices
    valid_ranges = ranges[indValid]
    valid_angles = angles[indValid]

    # xy position in the physical frame
    xs = np.array([valid_ranges * np.cos(valid_angles)])
    ys = np.array([valid_ranges * np.sin(valid_angles)])
    location = np.concatenate([np.concatenate([xs, ys], axis=0), np.zeros((1, xs.shape[1]))], axis=0)
    location = np.concatenate([location, np.ones((1, xs.shape[1]))], axis=0)  # 4*1081
    location = np.dot(bTl, location)

    # delta is relative pose change
    if i == 0:
        delta = lidar_pose[0].reshape(3, 1)
    else:
        delta = sum(lidar_pose[i-(step-1):i+1]).reshape(3, 1).astype(np.float)

    # Particle Filter predict and update
    particle = PF.predict(N, delta, particle)
    particle, weight = PF.update(N, MAP, location, particle, weight, x_im, y_im, x_range, y_range)

    # Find the best particle and its trajectory
    max_weight_idx = np.argmax(weight)
    best_particle = particle[:, max_weight_idx]  # 3*1
    trajectory = np.hstack((trajectory, best_particle[0:2].reshape(2, 1)))

    # Map update
    MAP = mapping.update(MAP, best_particle, location)

    # Particle Filter resample
    Neff = 1 / np.dot(weight.reshape(1, N), weight.reshape(N, 1)).squeeze()
    if Neff < 5:
        particle, weight = PF.resample(N, particle, weight)

    # Test
    if i%500 == 0:
        print(i)

# MAP['map'] = ((1 - 1 / (1 + np.exp(MAP['map']))) > 0.8).astype(np.int)
trajectory_x = np.ceil((trajectory[0] - MAP['xmin']) / MAP['res']).astype(np.int16) - 1
trajectory_y = np.ceil((trajectory[1] - MAP['xmin']) / MAP['res']).astype(np.int16) - 1

plt.figure(figsize=(8, 6), dpi=300)
a = plt.imshow(MAP['map'])
plt.plot(trajectory_y[0:-1:10], trajectory_x[0:-1:10], color="red", linewidth=1, linestyle="-")
plt.savefig('trainset_test.png')
plt.show()

plt.figure()
plt.plot(trajectory_x, trajectory_y, color="blue", linewidth=1.0, linestyle="-")
plt.plot(trajectory_x[0:-1:10], trajectory_y[0:-1:10], color="blue", linewidth=1.0, linestyle="-")
plt.savefig('track_test.png')
plt.show()

