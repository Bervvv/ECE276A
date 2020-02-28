import p2_utils as p2
import numpy as np


def update(MAP, best_particle, location):
    # wTb of the best particle
    best_wTb = np.array([[np.cos(best_particle[2]), -np.sin(best_particle[2]), 0, best_particle[0]],
                         [np.sin(best_particle[2]), np.cos(best_particle[2]), 0, best_particle[1]],
                         [0, 0, 1, 0.93], [0, 0, 0, 1]])

    # start point sx and sy in image
    sx = np.ceil((best_particle[0] - MAP['xmin']) / MAP['res']).astype(np.int16) - 1
    sy = np.ceil((best_particle[1] - MAP['ymin']) / MAP['res']).astype(np.int16) - 1

    endpoint = np.dot(best_wTb, location)
    bool_endpoint = endpoint[2, :] >= 0.01  # remove the points that hit the ground
    endpoint = endpoint[:, bool_endpoint]

    # end point ex and ey in image
    ex = np.ceil((endpoint[0] - MAP['xmin']) / MAP['res']).astype(np.int16) - 1
    ey = np.ceil((endpoint[1] - MAP['ymin']) / MAP['res']).astype(np.int16) - 1

    for i in range(len(ex)):
        # bresenham2D
        points = p2.bresenham2D(sx, sy, ex[i], ey[i])
        [xis, yis] = points.astype(np.int16)
        indGood = np.logical_and(np.logical_and(np.logical_and((xis > 1), (yis > 1)), (xis < MAP['sizex'])),
                                 (yis < MAP['sizey']))

        # To simplify
        MAP['map'][ex[i], ey[i]] = 1
        MAP['map'][xis[indGood], yis[indGood]] = 1

    #     MAP['map'][xis[indGood], yis[indGood]] += np.log(1 / 4)
    #     MAP['map'][ex[i], ey[i]] += np.log(4)
    # MAP['map'] = np.clip(MAP['map'], 5 * np.log(1 / 4), 5 * np.log(4)) # Avoid over confidence
    return MAP
