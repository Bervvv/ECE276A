from roipoly import roipoly
import matplotlib
matplotlib.use('Qt5Agg')
import cv2
import pylab as pl
import numpy as np
import pickle
import os

Fake_red = [[], [], []]
root = os.listdir('/Users/hebolin/Desktop/ECE276A_PR1/trainset')
del root[2]

for i in root:
    image = cv2.imread('./trainset/' + i)
    image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # converted color space
    pl.imshow(image2)
    roi = roipoly(color='r')

    # Displaying a ROI
    pl.imshow(image2)
    roi.display_roi()
    pl.show()

    # Extracting a binary mask image
    mask = roi.get_mask(image2)
    pl.imshow(mask)
    pl.show()

    # Labelling
    position = np.where(mask == True)
    image3 = cv2.cvtColor(image2, cv2.COLOR_RGB2YCR_CB)  # converted color space

    Y = image3[:, :, 0]
    CR = image3[:, :, 1]
    CB = image3[:, :, 2]

    Y_label = Y[position]
    CR_label = CR[position]
    CB_label = CB[position]

    Fake_red[0].extend(Y_label.tolist())
    Fake_red[1].extend(CR_label.tolist())
    Fake_red[2].extend(CB_label.tolist())

# Save data
Fake_red_output = open('Fake_red.pkl', 'wb')
pickle.dump(Fake_red, Fake_red_output)
Fake_red_output.close()

