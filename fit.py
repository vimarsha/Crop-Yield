from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
import keras
from keras import regularizers, optimizers
from keras.layers import Conv2D,Input,Dense,MaxPooling2D,BatchNormalization,ZeroPadding2D,Flatten,Dropout
from keras.models import Model
import numpy as np
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau, CSVLogger,EarlyStopping,ModelCheckpoint
import matplotlib.pyplot as plt
from scipy import misc
from numpy.random import permutation

def le_net():
    model = Sequential()
    # first set of CONV => RELU => POOL
    model.add(Convolution2D(20, 5, 5, border_mode="same",input_shape=(60,60,3)))
    model.add(Dropout(0.2))
    model.add(Activation("relu"))
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering="th"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # second set of CONV => RELU => POOL
    model.add(Convolution2D(50, 5, 5, border_mode="same"))
    model.add(Dropout(0.2))
    model.add(Activation("relu"))
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering="th"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Dropout(0.2))
    model.add(Activation("relu"))
 
    # softmax classifier
    model.add(Dense(10))
    model.add(Dropout(0.2))
    model.add(Activation("softmax"))
        
    return model

model = le_net()
model.summary()

train_images = np.load('train_images_lenet.npy')
train_labels = np.load('train_labels_lenet.npy')
#print(train_images[0])
#img=plt.imshow(train_images[0].astype('uint8'))
#print(img)

'''lr_reducer = ReduceLROnPlateau(factor = np.sqrt(0.1), cooldown=0, patience=2, min_lr=0.5e-6)
csv_logger = CSVLogger('Lenet.csv')
early_stopper = EarlyStopping(min_delta=0.001,patience=30)
model_checkpoint = ModelCheckpoint('Lenet.hdf5',monitor = 'val_loss', verbose = 1,save_best_only=True)
'''
# Test pretrained model
train_images = np.array(train_images)
train_labels = np.array(train_labels)
mean = np.mean(train_images,axis=(0,1,2,3))
std = np.std(train_images,axis=(0,1,2,3))
#train_images = (train_images-mean)/(std+1e-7)
num_classes = 10
train_labels = np_utils.to_categorical(train_labels,num_classes)


perm = permutation(len(train_images))
train_images = train_images[perm]
train_labels = train_labels[perm]
val_images = train_images[1:4800]
val_labels = train_labels[1:4800]
new_train= train_images[4800:]
new_labels = train_labels[4800:]

print(perm)

lr_reducer = ReduceLROnPlateau(factor = np.sqrt(0.1), cooldown=0, patience=2, min_lr=0.5e-6)
csv_logger = CSVLogger('Lenet.csv')
early_stopper = EarlyStopping(min_delta=0.001,patience=30)
model_checkpoint = ModelCheckpoint('Lenet.hdf5',monitor = 'val_loss', verbose = 1,save_best_only=True)


model.compile(loss='categorical_crossentropy',
        optimizer="Adam",
        metrics=['accuracy'])

# This will do preprocessing and realtime data augmentation:
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
  
datagen.fit(new_train)
model.fit(datagen.flow(new_train, new_labels, batch_size=12),
                        epochs=3,verbose=1,validation_data=(val_images,val_labels))
#model.save('vim2.model')

#Training the model 
model.save_weights('lenet.h5')

model.save('vim3.model')

'''
#Loading pre-trained weights
model.load_weights('lenet.h5')

class_names = {0:'bacterial_spot',1:'healthy',2:'late_blight',3:'leaf_mold',4:'septorial_leaf_spot',5:'mosaic_virus',6:'curl_virus'}

img = misc.imresize(misc.imread("c.jpg", mode='RGB'),(60,60)).astype(np.float32)
img = (img-mean)/(std+1e-7)
img = np.expand_dims(img, axis=0)
out = model.predict(img) 
print (out)
print (np.argmax(out))
print(class_names[np.argmax(out)])
'''

'''
img = misc.imresize(misc.imread("Tomato/Tomato___healthy/0a9986e6-b629-4ff5-8aab-7488ea9b935b___RS_HL 9704.JPG", mode='RGB'),(60,60)).astype(np.float32)
img = (img-mean)/(std+1e-7)
img = np.expand_dims(img, axis=0)
out = model.predict(img) 
print( out)
print (np.argmax(out))
print(class_names[np.argmax(out)])

labels = np.load('train_labels_lenet.npy')
labels_list = list(labels) 
print(type(labels_list))
print(len(labels_list))

for i in range(10):
    print(str(i)+" "+str(labels_list.count(i)))'''