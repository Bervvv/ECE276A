import numpy as np
import p2_utils as p2


def softmax(x):
    ex = np.exp(x - np.max(x))
    return ex / ex.sum()


def predict(N, delta, particle):
    mean = 0
    variance = 0.05
    particle += np.tile(delta, N) + np.array([np.random.normal(mean, variance, N),
                                             np.random.normal(mean, variance, N),
                                             np.random.normal(mean, variance, N)])
    return particle


def update(N, MAP, location, particle, weight, x_im, y_im, x_range, y_range):
    correlation = np.zeros(N)  # 1 * N
    # To simplify
    # temp = ((1 - 1 / (1 + np.exp(MAP['map']))) > 0.8).astype(np.int)
    for i in range(N):
        xw = particle[:, i][0]
        yw = particle[:, i][1]
        thetaw = particle[:, i][2]
        wTb = np.array([[np.cos(thetaw), -np.sin(thetaw), 0, xw],
                        [np.sin(thetaw), np.cos(thetaw), 0, yw],
                        [0, 0, 1, 0.93], [0, 0, 0, 1]])
        endpoint = np.dot(wTb, location)
        bool_endpoint = endpoint[2, :] >= 0.01  # remove the points that hit the ground
        endpoint = endpoint[:, bool_endpoint]
        count = p2.mapCorrelation(MAP['map'], x_im, y_im, endpoint[0:2], x_range, y_range)
        # To simplify
        # count = p2.mapCorrelation(temp, x_im, y_im, endpoint[0:2], x_range, y_range)
        correlation[i] = np.max(count)

        # update particle
        max_idx = np.unravel_index(count.argmax(), count.shape)
        particle[0, i] += (max_idx[0]-4) * MAP['res']
        particle[1, i] += (max_idx[1]-4) * MAP['res']

    # update weight
    ph = softmax(correlation)
    weight = weight * ph / np.sum(weight * ph)
    return particle, weight


def resample(N, particle, weight):
    new_particle = np.zeros((3, N))
    new_weight = np.ones([1, N]) * (1 / N)
    j = 0
    c = weight.squeeze()[0]
    for k in range(N):
        u = np.random.uniform(0, 1 / N)
        beta = u + k / N
        while beta > c:
            j = j + 1
            c = c + weight.squeeze()[j]
        new_particle[:, k] = particle[:, j]
    return new_particle, new_weight
