# Code reference from https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

image1 = cv.imread('Query.png', cv.IMREAD_GRAYSCALE)
image2 = cv.imread('Train.png', cv.IMREAD_GRAYSCALE)

# Initiate SIFT detector
sift = cv.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(image1, None)
kp2, des2 = sift.detectAndCompute(image2, None)

# create BFMatcher with default parameters
bf = cv.BFMatcher()

# Match descriptors
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

# Draw first 10 matches
image3 = cv.drawMatchesKnn(image1,kp1,image2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.imshow(image3), plt.show()