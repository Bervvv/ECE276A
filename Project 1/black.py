from roipoly import roipoly
import matplotlib
matplotlib.use('Qt5Agg')
import cv2
import pylab as pl
import numpy as np
from skimage.measure import label, regionprops
import pickle
import os

black = [[], [], []]

root = os.listdir('/Users/hebolin/Desktop/ECE276A_PR1/trainset')
del root[2]

for i in root:
    image = cv2.imread('./trainset/' + i)
    image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # converted color space
    pl.imshow(image2)
    roi = roipoly(color='r')
    mask = roi.get_mask(image2)

    # Labelling
    position = np.where(mask == True)
    image3 = cv2.cvtColor(image2, cv2.COLOR_RGB2YCR_CB)  # converted color space

    Y = image3[:, :, 0]
    CR = image3[:, :, 1]
    CB = image3[:, :, 2]

    Y_label = Y[position]
    CR_label = CR[position]
    CB_label = CB[position]

    black[0].extend(Y_label.tolist())
    black[1].extend(CR_label.tolist())
    black[2].extend(CB_label.tolist())

    label_mask = label(mask)
    region = regionprops(label_mask)

# Save data
black_output = open('black.pkl', 'wb')
pickle.dump(black, black_output)
black_output.close()







