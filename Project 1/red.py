from roipoly import roipoly
import matplotlib
matplotlib.use('Qt5Agg')
import cv2
import pylab as pl
import numpy as np
from skimage.measure import label, regionprops
import pickle
import os

Red = [[], [], []]
# Red_area = []
# Red_area_ratio = []

root = os.listdir('/Users/hebolin/PycharmProjects/ece271_hw1/trainset')
del root[2]

for i in root:
    image = cv2.imread('./trainset/' + i)
    image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # converted color space
    pl.imshow(image2)
    roi = roipoly(color='r')

    # # Displaying a ROI
    # pl.imshow(image2)
    # roi.display_roi()
    # pl.show()

    # Extracting a binary mask image
    mask = roi.get_mask(image2)
    # pl.imshow(mask)
    # pl.show()

    # Labelling
    position = np.where(mask == True)
    image3 = cv2.cvtColor(image2, cv2.COLOR_RGB2YCR_CB)  # converted color space

    Y = image3[:, :, 0]
    CR = image3[:, :, 1]
    CB = image3[:, :, 2]

    Y_label = Y[position]
    CR_label = CR[position]
    CB_label = CB[position]

    Red[0].extend(Y_label.tolist())
    Red[1].extend(CR_label.tolist())
    Red[2].extend(CB_label.tolist())

    label_mask = label(mask)
    region = regionprops(label_mask)
    # Red_area.append(region[0].area)
    # Red_area_ratio.append(region[0].area/(image.shape[0]*image.shape[1]))

# Save data
red_output = open('red.pkl', 'wb')
pickle.dump(Red, red_output)
red_output.close()

# red_area_output = open('red_area.pkl', 'wb')
# pickle.dump(Red_area, red_area_output)
# red_area_output.close()
#
# red_area_ratio_output = open('red_area_ratio.pkl', 'wb')
# pickle.dump(Red_area_ratio, red_area_ratio_output)
# red_area_ratio_output.close()





