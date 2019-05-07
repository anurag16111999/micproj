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
mypath = "/home/mohan_abhyas/Documents/acads/cs736/micproj/imclass/chest_xray/train/" + folder
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

# print(onlyfiles)
i1 = 1
seltol = 800
#seltol = 10
# selected1 = 800
# for selected1 in [200]:
samplpositive = 500
#samplpositive = 10
featurestotal = []
for image in onlyfiles:
	if(image[-4:] != "jpeg"):
		continue
	i1 = i1 + 1
	im1 = mypath+"/" + image
	im = imread(im1,cv2.COLOR_BGR2GRAY)
	res = cv2.resize(im, dsize=(250,250), interpolation=cv2.INTER_CUBIC)
	#res = cv2.resize(im, dsize=(500,500))
	k1 = np.reshape(res,-1)
        k1 = list(k1)
	# print(k1.shape)
        print(i1)
	featurestotal.append(k1)
	if(i1 == seltol + 1):
		break

print("setp1")

folder = "PNEUMONIA"
mypath = "/home/mohan_abhyas/Documents/acads/cs736/micproj/imclass/chest_xray/train/" + folder
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

# featurestotal = []
i2 = 1
for image in onlyfiles:
	if(image[-4:] != "jpeg"):
		continue
	i2 = i2 + 1
	im1 = mypath+"/" + image
	im = imread(im1,cv2.COLOR_BGR2GRAY)
	res = cv2.resize(im, dsize=(250,250), interpolation=cv2.INTER_CUBIC)
	k1 = np.reshape(res,-1)
        k1 = list(k1)
	# print(k1.shape)
	featurestotal.append(k1)
        print(i2)
	if(i2 == samplpositive + 1):
		break
	# 


# featurestotal1 = StandardScaler().fit_transform(featurestotal)
# print(featurestotal1)
#featurestotal = np.array(featurestotal
# print(featurestotal)
#print(featurestotal.shape)
	
	# selecting 100 features to train
print("Nippu ra")
sc = StandardScaler()
for selected1 in [400]:
	features111 = sc.fit_transform(featurestotal[:selected1])
        features211 = sc.transform(featurestotal[selected1:])
        featurestotal = []
	pca = PCA(n_components = 100)
        xtr = pca.fit_transform(features111)
        xte = pca.transform(features211)

	csvdata = list(xtr)
        features111 = []
        csvdata.extend(xte)
        features211 = []
        for i in range(len(csvdata)):
            if i >= seltol:
                csvdata[i] = list(csvdata[i])
                csvdata[i].append(1)
            else:
                csvdata[i] = list(csvdata[i])
                csvdata[i].append(0)
            i = i+1

	with open("outputfull" + str(selected1) + ".csv", "wb") as f:
	    writer = csv.writer(f)
	    writer.writerows(csvdata)
