from keras.layers import Input, Dense
from keras.models import Model
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
#seltol = 10
# selected1 = 800
# for selected1 in [200]:
samplpositive = 600
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
mypath = "/home/anurag/Documents/cs 736/chest-xray-pneumonia/chest_xray/chest_xray/train/" + folder
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


x_train = featurestotal[:600]
x_test = featurestotal[600:]

# this is the size of our encoded representations
encoding_dim = 100  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(62500,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(62500, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)
# Let's also create a separate encoder model:

# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)
# As well as the decoder model:

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))
# Now let's train our autoencoder to reconstruct MNIST digits.

# First, we'll configure our model to use a per-pixel binary crossentropy loss, and the Adadelta optimizer:

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
# Let's prepare our input data. We're using MNIST digits, and we're discarding the labels (since we're only interested in encoding/decoding the input images).

from keras.datasets import mnist
import numpy as np
# (x_train, _), (x_test, _) = mnist.load_data()
# We will normalize all values between 0 and 1 and we will flatten the 28x28 images into vectors of size 784.

x_train = np.array(x_train)
x_test = np.array(x_test)

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print x_train.shape
print x_test.shape
# Now let's train our autoencoder for 50 epochs:

autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))
# After 50 epochs, the autoencoder seems to reach a stable train/test loss value of about 0.11. We can try to visualize the reconstructed inputs and the encoded representations. We will use Matplotlib.

# encode and decode some digits
# note that we take them from the *test* set

csvdata = []

encoded_imgs = encoder.predict(x_train)
csvdata = [(xi.tolist() + [0]) for xi in encoded_imgs]

encoded_imgs = encoder.predict(x_test[:700])
csvdata.extend([(xi.tolist() + [0]) for xi in encoded_imgs])

encoded_imgs = encoder.predict(x_test[700:])
csvdata.extend([(xi.tolist() + [1]) for xi in encoded_imgs])

with open("outputautoencoder.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(csvdata)




# decoded_imgs = decoder.predict(encoded_imgs)
# # use Matplotlib (don't ask)
# import matplotlib.pyplot as plt

# n = 10  # how many digits we will display
# plt.figure(figsize=(20, 4))
# for i in range(n):
#     # display original
#     ax = plt.subplot(2, n, i + 1)
#     plt.imshow(x_test[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)

#     # display reconstruction
#     ax = plt.subplot(2, n, i + 1 + n)
#     plt.imshow(decoded_imgs[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()		