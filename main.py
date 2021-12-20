import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from matplotlib.image import imread
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
import pandas as pd
import time
import os

X = []
Y = []
dir = 'train/'
for imgName in os.listdir(dir):
    img = imread(dir + imgName)
    X.append(imgName)
    if imgName[0] == 'c':
        Y.append(0)
    else:
        Y.append(1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0, shuffle=True)

xTest = []
yTest = []
cnt = 0
for imgName in X_test:
    img = imread('train/' + imgName)
    resized_img = resize(img, (128, 64))
    fd, hog_img = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True,
                      multichannel=True)
    xTest.append(fd)
    if imgName[0] == 'c':
        yTest.append(0)
    else:
        yTest.append(1)



xTrain = []
yTrain = []
cnt = 0
for imgName in X_train:
    img = imread(dir + imgName)
    resized_img = resize(img, (128, 64))
    fd, hog_img = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True,
                      multichannel=True)
    xTrain.append(fd)
    if imgName[0] == 'c':
        yTrain.append(0)
    else:
        yTrain.append(1)
    cnt += 1



startTime = time.time()
print('Training Model Starts ...')
svm_kernel_ovr = OneVsRestClassifier(SVC(kernel='linear', C=1)).fit(xTrain, yTrain)
svm_kernel_ovo = SVC(kernel='linear', C=1).fit(xTrain, yTrain)

svm_linear_ovr = LinearSVC().fit(xTrain, yTrain)
svm_linear_ovo = OneVsOneClassifier(LinearSVC()).fit(xTrain, yTrain)
print('Training Model ends')
endTime = time.time()
print('training time:' + str(endTime - startTime) + 's')


accuracy = svm_kernel_ovr.score(xTest, yTest)
print('Kernel OneVsRest SVM accuracy: ' + str(accuracy))
accuracy = svm_kernel_ovo.score(xTest, yTest)
print('Kernel OneVsOne SVM accuracy: ' + str(accuracy))