import os, cv2
import numpy as np
from skimage.measure import label, regionprops

class StopSignDetector():

    def __init__(self):
        self.mean_red = [[96.06214209877389], [195.98988971847388], [110.16503442016577]]
        self.mean_yellow = [[153.4965087529189], [164.0425092718365], [76.24108873490943]]
        self.mean_black = [[39.90913573962426], [125.14013012423842], [131.4960628668692]]
        self.mean_fake_red = [[140.30234245338872], [122.31818689157842], [121.58589481040055]]

        self.cov_red = [[  3232.40911135, -11132.20292696,   -186.67559589],
                       [-11132.20292696,   1360.52123188,  -7647.95013781],
                       [  -186.67559589,  -7647.95013781,    150.38283229]]
        self.cov_blue = [[ 5.67506576e+02, -5.58896348e+00, -5.72622245e+03],
                       [-5.58896348e+00,  5.21579559e+01, -5.95613524e+03],
                       [-5.72622245e+03, -5.95613524e+03,  6.93624323e+01]]
        self.cov_yellow = [[  979.68959055,  -112.29872421, -6447.34687934],
                       [ -112.29872421,    72.47298057, -7755.79175697],
                       [-6447.34687934, -7755.79175697,   510.43182111]]
        self.cov_black = [[  531.69608004, -7263.92179451, -8398.71807693],
                       [-7263.92179451,    18.09386583,   -49.49034095],
                       [-8398.71807693,   -49.49034095,    17.77832038]]
        self.cov_white = [[   468.342636  , -12096.99969677, -11337.38300055],
                       [-12096.99969677,     15.60089749,    -39.49060832],
                       [-11337.38300055,    -39.49060832,     35.10851416]]
        self.cov_fake_red = [[ 3074.48732956,   391.05814379,  -734.2280698 ],
                       [  391.05814379,  1328.30199029, -1002.63908271],
                       [ -734.2280698 , -1002.63908271,  1036.21677167]]


    def probability(self, mean, cov, data):
        val = np.dot(-0.5 * (data - mean).T, np.linalg.inv(cov))
        val2 = (data - mean)
        val3 = np.zeros(data.shape[1])
        for i in range(data.shape[1]):
            val3[i] = np.dot(val[i, :], val2[:, i])
        val4 = (2 * np.pi) ** 3 * abs(np.linalg.det(cov))
        p = 1 / (val4 ** 0.5) * np.exp(val3)
        return p


    def segment_image(self, img):
        image2 = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        x,y,z = image2.shape
        ratio = 1
        resize_img = image2
        image3 = resize_img.reshape(int(x/ratio)*int(y/ratio),3)
        pixel = np.matrix(np.transpose([image3]))
        p_red = self.probability(self.mean_red, self.cov_red, pixel)
        p_fake_red = self.probability(self.mean_fake_red, self.cov_fake_red, pixel)
        p_yellow = self.probability(self.mean_yellow, self.cov_yellow, pixel)
        p_black = self.probability(self.mean_black, self.cov_black, pixel)

        collection = (p_red ,p_fake_red,p_yellow,p_black)
        collection2 = np.array(collection)
        bool_collection2 = (np.max(collection2,0) == collection2[0,:])
        position = np.where(bool_collection2==True,1,0)
        mask_img = position.reshape(int(x/ratio),int(y/ratio))
        return mask_img


    def get_bounding_box(self, img):
        mask_img = self.segment_image(img)
        label_mask = label(mask_img, connectivity=1)
        region = regionprops(label_mask)
        x,y = mask_img.shape
        box_index = []
        ratio = 1

        for i in range(len(region)):
            if region[i].extent >0.3 and region[i].extent < 0.7 and region[i].eccentricity>0.3 \
                    and region[i].eccentricity<0.7  and region[i].area/(x*y) > 0.003 and region[i].area > 45:
                box_index.append(i)
                print(region[i].extent, region[i].eccentricity, region[i].area/(x*y), region[i].area)
        boxes = []
        # 0.3,0.7,0.3,0.7,0.003,45

        for j in range(len(box_index)):
            xy = region[j].bbox
            boxes.append([ratio*xy[1], ratio*(x-xy[2]), ratio*(xy[3]),ratio*(x-xy[0])])
        print(boxes)
        return boxes


if __name__ == '__main__':
    folder = "trainset"
    my_detector = StopSignDetector()
    for filename in os.listdir(folder):
        # read one test image
        img = cv2.imread(os.path.join(folder,filename))
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()