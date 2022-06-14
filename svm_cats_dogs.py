from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from imutils import paths
import numpy as np
#import argparse
import imutils
import cv2
import os
import _pickle as cPickle

def extract_color_histogram(image, bins=(8, 8, 8)):
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
		[0, 180, 0, 256, 0, 256])

	if imutils.is_cv2():
		hist = cv2.normalize(hist)
	else:
		cv2.normalize(hist, hist)

	return hist.flatten()

imagePaths = list(paths.list_images('D:/level3/semster2/selected2/dataset/train1'))
data = []
labels = []

for (i, imagePath) in enumerate(imagePaths):
	image = cv2.imread(imagePath)
	label = imagePath.split(os.path.sep)[-1].split(".")[0]

	hist = extract_color_histogram(image)
	data.append(hist)
	labels.append(label)

	if i > 0 and i % 100 == 0:
		print("{}/{}".format(i, len(imagePaths)))

le = LabelEncoder()
labels = le.fit_transform(labels)

(trainData, testData, trainLabels, testLabels) = train_test_split(
	np.array(data), labels, test_size=0.25, random_state=42)

model = LinearSVC()
model.fit(trainData, trainLabels)

predictions = model.predict(testData)
print(classification_report(testLabels, predictions,
	target_names=le.classes_))

f = open("svm_model.cpickle", "wb")
f.write(cPickle.dumps(model))
f.close()

#model = cPickle.loads(open('model.cpickle', "rb").read())

singleImage = cv2.imread('D:/level3/semster2/selected2/dataset/test1/1383.jpg') 
histt = extract_color_histogram(singleImage)
histt2 = histt.reshape(1, -1)
prediction = model.predict(histt2)
if(prediction==0):
     print(" i guess it cat pic")
else:
    print("i guess it dog pic")
