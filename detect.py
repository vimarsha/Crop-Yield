
from scipy import misc
import numpy as np
import tensorflow as tf
#from numpy.random import permutation

class_names = {0:'bacterial_spot',1:'healthy',2:'late_blight',3:'leaf_mold',4:'septorial_leaf_spot',5:'mosaic_virus',6:'curl_virus'}

a = np.load('train_images_lenet.npy')
b = np.load('train_labels_lenet.npy')
a = np.array(a)
b = np.array(b)
mean = np.mean(a,axis=(0,1,2,3))
std = np.std(a,axis=(0,1,2,3))

model=tf.keras.models.load_model("vim3.model")
model.load_weights('lenet.h5')

img = misc.imresize(misc.imread("c.jpg", mode='RGB'),(60,60)).astype(np.float32)
#img = imageio.imread("h.png")
#img=img.resize(60,60).astype(np.float32)
img = ((img-mean)/(std+1e-7))
img = np.expand_dims(img, axis=0)
out = model.predict(img) 
print (out)
print (np.argmax(out))
print(class_names[np.argmax(out)])