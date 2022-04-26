# Code reference from https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

image1 = cv.imread('Query.png', cv.IMREAD_GRAYSCALE)
image2 = cv.imread('Train.png', cv.IMREAD_GRAYSCALE)

# Initiate ORB detector
orbd = cv.ORB_create()

# find the keypoints and descriptors with ORB
kp1, des1 = orbd.detectAndCompute(image1, None)
kp2, des2 = orbd.detectAndCompute(image2, None)

# create BFMatcher object
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

# Match descriptors
matches = bf.match(des1, des2)

# Sort them in the order of their distance
matches = sorted(matches, key=lambda x: x.distance)

# Draw first 10 matches
image3 = cv.drawMatches(image1, kp1, image2, kp2, matches[:100], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.imshow(image3), plt.show()