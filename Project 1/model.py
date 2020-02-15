import pickle
import numpy as np
import math
from numpy import *


# Define Mixture Gaussian Model based on class
def gmm(data, val2=0):
    # Calculate mean
    Y_mean = sum(data[0]) / len(data[0])
    CR_mean = sum(data[1]) / len(data[1])
    CB_mean = sum(data[2]) / len(data[2])
    mean = [[Y_mean], [CR_mean], [CB_mean]]

    # Calculate covariant
    sample = np.array(data)
    x, y = sample.shape
    for i in range(y):
        val = (sample[:, i] - mean) * (sample[:, i] - mean).T  # find number of column
        val2 += val
    cov = val2 / (y - 1)
    return mean, cov


# Define probability based on [3,1] pixel
def probability(mean, cov, data):
    val = -0.5 * (data - mean).T * np.linalg.inv(cov) * (data - mean)
    val2 = (2 * np.pi) ** 3 * abs(np.linalg.det(cov))
    p = 1 / (val2 ** 0.5) * np.exp(val)
    return p

#
# def probability2(mean, cov, data):
#         '''gaussian probability for one 3*1 vector'''
#         val = -0.5 * (data - mean).T * np.linalg.inv(cov) * (data - mean)
#         A_D = abs(np.linalg.det(cov)) ** 0.5
#         P = 1 / (A_D * (2 * math.pi) ** 1.5) * math.exp(val)
#         return P

# def probability3(mean, cov, data):
#     det = abs(np.linalg.det(cov))
#     norm_const = 1.0 / (math.pow((2 * np.pi), 3 / 2) * math.pow(det, 1 / 2))
#     x_mu = np.matrix(data - np.matrix(mean))
#     inv = np.linalg.inv(cov)
#     result = math.pow(math.e, -0.5 * (x_mu.T * inv * x_mu))
#     return norm_const * result
    # print(probability2(mean, cov, data))


#

# Load data from pkl file
red_input = open('red.pkl', 'rb')
red = pickle.load(red_input)
red_input.close()

# fake_red_input = open('Fake_red.pkl', 'rb')
# fake_red = pickle.load(fake_red_input)
# fake_red_input.close()
#
# red_area_input = open('red_area.pkl', 'rb')
# red_area = pickle.load(red_area_input)
# red_area_input.close()
#
# blue_input = open('blue.pkl', 'rb')
# blue = pickle.load(blue_input)
# blue_input.close()
#
# yellow_input = open('yellow.pkl', 'rb')
# yellow = pickle.load(yellow_input)
# yellow_input.close()
#
# green_input = open('green.pkl', 'rb')
# green = pickle.load(green_input)
# green_input.close()
#
# white_input = open('white.pkl', 'rb')
# white = pickle.load(white_input)
# white_input.close()
#
# black_input = open('black.pkl', 'rb')
# black = pickle.load(black_input)
# black_input.close()
#
# grey_input = open('grey.pkl', 'rb')
# grey = pickle.load(grey_input)
# grey_input.close()
#
# random_input = open('random.pkl', 'rb')
# random = pickle.load(random_input)
# random_input.close()

# Calculate parameters
[mean_red, cov_red] = gmm(red)
# [mean_green, cov_green] = gmm(green)
# [mean_blue, cov_blue] = gmm(blue)
# [mean_white, cov_white] = gmm(white)
# [mean_grey, cov_grey] = gmm(grey)
# [mean_black, cov_black] = gmm(black)
# [mean_yellow, cov_yellow] = gmm(yellow)
# [mean_fake_red, cov_fake_red] = gmm(fake_red)
# [mean_random, cov_random] = gmm(random)
