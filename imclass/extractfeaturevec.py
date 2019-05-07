import csv
from os import listdir
from os.path import isfile, join
from scipy.misc import imread
import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from numpy import linalg as LA
import matplotlib.pyplot as plt


folder = "NORMAL"
mypath = "/home/anurag/Documents/cs 736/chest-xray-pneumonia/chest_xray/chest_xray/train/" + folder
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

# print(onlyfiles)
i1 = 1
seltol = 1300
# selected1 = 800
# for selected1 in [200]:
samplpositive = 1000
featurestotal = []
for image in onlyfiles:
	if(image[-4:] != "jpeg"):
		continue
	i1 = i1 + 1
	im1 = folder + "/" + image
	im = imread(im1,cv2.COLOR_BGR2GRAY)
	res = cv2.resize(im, dsize=(50,50), interpolation=cv2.INTER_CUBIC)
	# z = im.shape;
	# print(im.shape)
	# k1 = np.reshape(im,z[0]*z[1])
	k1 = np.reshape(res,-1)
	# print(k1.shape)
	featurestotal.append(k1)
	if(i1 == seltol + 1):
		break

folder = "PNEUMONIA"
mypath = "/home/anurag/Documents/cs 736/chest-xray-pneumonia/chest_xray/chest_xray/train/" + folder
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

# featurestotal = []
i2 = 1
for image in onlyfiles:
	if(image[-4:] != "jpeg"):
		continue
	i2 = i2 + 1
	im1 = folder+ "/" + image
	im = imread(im1,cv2.COLOR_BGR2GRAY)
	res = cv2.resize(im, dsize=(50,50), interpolation=cv2.INTER_CUBIC)
	# print(image)
	# z = im.shape;
	# print(im.shape)
	# k1 = np.reshape(im,z[0]*z[1])
	# print(res.shape)
	k1 = np.reshape(res,-1)
	# print(k1.shape)
	featurestotal.append(k1)
	if(i2 == samplpositive + 1):
		break
	# 


# featurestotal1 = StandardScaler().fit_transform(featurestotal)
# print(featurestotal1)
featurestotal = np.array(featurestotal)
# print(featurestotal)
print(featurestotal.shape)
	
	# selecting 100 features to train
for selected1 in [100,200,400,800,1200]:

	features = StandardScaler().fit_transform(featurestotal[:selected1])
	# print(featurestotal[:100].shape)

	feamean = np.mean(featurestotal,axis = 0)

	feastd = np.std(featurestotal,axis = 0)

	# print(k1)
	# for image in onlyfiles:

	# features = featurestotal

	# exit()
	cov = np.dot(np.transpose(features),features)
	# print(features)
	print(cov.shape)

	eigenValues, eigenVectors = LA.eig(cov)


	idx = eigenValues.argsort()[::-1]   
	eigenValues = eigenValues[idx]
	eigenVectors = eigenVectors[:,idx]
	print(np.dot(np.transpose(eigenVectors),eigenVectors))
	# print(eigenValues)
	# print(eigenVectors)
	# plt.plot(eigenValues[:50])
	# plt.show()



	csvdata = []
	for image in features:
		# print(image.shape)
		data = []
		for eigv in eigenVectors[:50]:
			# print(eigv.shape)
			val1 = np.dot(image,eigv)
			# print(val1.shape)
			data.append(val1.real)
		csvdata.append(data + [0])

	# with open("output.csv" + "data", "wb") as f:
	#     writer = csv.writer(f)
	#     writer.writerows(csvdata)
	# csvdata = []

	# print(len(csvdata))
	# print(featurestotal.shape)
	k2 = 0
	for image in featurestotal[selected1:]:
		# print(image.shape)
		image = image - feamean
		image = image/feastd
		k2 = k2+1
		data = []
		for eigv in eigenVectors[:50]:
			# print(eigv.shape)
			val1 = np.dot(image,eigv)
			# print(val1.shape)
			data.append(val1.real)
		if(k2 <= (seltol - selected1)):
			csvdata.append(data + [0])
		else:
			csvdata.append(data + [1])
	# 
	# csvdata = eigenVectors[:20]
	# print(len(ccs))

	exit()
	with open("outputfull" + str(selected1) + ".csv", "wb") as f:
	    writer = csv.writer(f)
	    writer.writerows(csvdata)
