import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

def test_magnetogram(path_d):
    file_path = os.path.join(path_d, 'hmi_m_45s_2011_01_01_00_41_15_tai_magnetogram.png')
    # load the image
    img = cv2.imread(file_path, -1)
    img_p = plt.imread(file_path)
    print(img_p.shape)
    print(img.shape)
    # convert to grayscale
    # img[:, :, 3] = 
    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    cv2.imshow('gray', gray)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    print(gray[500, 500])
    print(gray.shape)
    print(gray[0, 0])
    # show the image
    # fig, ax = plt.subplots()
    # plt.imshow(gray, cmap='gray', vmin=0, vmax=255)
    # plt.show()


if __name__ == '__main__':
    path_d = 'data/noaa/magnetogram/2011/'
    test_magnetogram(path_d)