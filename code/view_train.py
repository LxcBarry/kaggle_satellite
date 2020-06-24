
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mimg
import numpy as np
import os
import cv2 as cv
from PIL import Image

# read data info from csv
train_root = "../input/train_images"
train_csv = "../input/train.csv"
img_num = 9


# color the triangle
def color_pic(pic_name,points):
    if points is np.nan:
        return
    pic = cv.imread(os.path.join(train_root,pic_name),cv.IMREAD_COLOR)
    double = points[0::2]
    odd = points[1::2]


    for point in zip(double,odd):
        for i in range(point[1]):
            if i % 20 == 0:
                pic = cv.circle(pic,(point[0]//1400,(point[0]+i)%1400),1,(0,0,255),2,0)

    return pic

if __name__ == "__main__":
    train_info = pd.read_csv(train_csv)
    print(train_info.head())
    pic_names = train_info["Image_Label"].values
    points_arr = train_info["EncodedPixels"].values
    points_arr = [list(map(int,x.split(' '))) if isinstance(x,str) else x for x in points_arr]

    #
    if img_num is not None:
        for i in range(img_num):
            pic = color_pic(pic_names[i].split('_')[0], points_arr[i])
            if pic is not None:
                # plt.subplot(3,3,i+1)
                # plt.imshow(pic)
                cv.namedWindow(str(i+1), cv.WINDOW_NORMAL)
                cv.imshow("img-"+pic_names[i].split('_')[1],pic)
                cv.waitKey(0)
                # cv.waitKey(10000)
                cv.destroyAllWindows()
                # Image.fromarray(pic).show()

    else:
        for pic_name,points in zip(pic_names,points_arr):
            color_pic(pic_name.split('_')[0],points)
    # plt.show()
    print('stop')

# plt.imshow()
# train_info.head()