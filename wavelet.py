import pywt
import cv2
import numpy as np


def w2d(img, mode="haar", level=1):
    imgArray = img
    imgArray = cv2.cvtColor(imgArray, cv2.COLOR_RGB2GRAY)  # cvt to gray
    imgArray = np.float32(imgArray)  # cvt to float
    imgArray /= 255

    # compute coefficients
    coeffs = pywt.wavedec2(imgArray, mode, level=level)

    # process coefficients
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0

    # reconstruction
    imgArray_H = pywt.waverec2(coeffs_H, mode)
    imgArray_H *= 255
    imgArray_H = np.uint8(imgArray_H)

    return imgArray_H
