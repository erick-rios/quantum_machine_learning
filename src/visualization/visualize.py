import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt


img = cv2.imread('../data/dogs_vs_cats/test/cats/cat.10000.jpg')

img.shape

plt.imshow(img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray.shape

gray

plt.imshow(gray, cmap='gray')

cat_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_eye.xml')

cats = cat_cascade.detectMultiScale(gray, 1.3, 5)
cats