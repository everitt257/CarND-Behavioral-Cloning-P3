# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 18:55:35 2017

@author: Xuandong Xu
"""

import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from keras.layers import Dense,Dropout,Flatten,Lambda,ELU
from keras.models import Sequential
from keras.layers.core import Activation
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
from sklearn.utils import shuffle


# function for generate batch image
def gen_batch_img(img_names, labels, batch_size):
    starti = 0
    endi = starti+batch_size
    n = len(img_names)
    while 1: 
        x_batch = np.array([mpimg.imread(image_name) for image_name in img_names[starti:endi]])
        y_batch = np.array(labels[starti:endi])
        starti += batch_size
        endi += batch_size
        if starti >= n:
            starti = 0
            endi = starti+batch_size
        # put images into cropping and resizing
        x_batch = np.array(pre_img(x_batch,64,64))
        yield (x_batch,y_batch)
        
# create model based on Nvidia's model
def create_model(learn_rate = 0.0001):
    # image normalization with lambda layer
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
                     input_shape=(64, 64, 3),
                     output_shape=(64, 64, 3)))
    #model.add(Input())

    #valid border mode should get rid of a couple each way, whereas same keeps
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid'))
    # Use ELU for more accurate result and speed up the training
    model.add(ELU())
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid'))
    model.add(ELU())
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid'))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid'))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid'))
    
    model.add(Flatten())
    # add in dropout of .5
    model.add(Dropout(.5))
    model.add(ELU())
    
    model.add(Dense(100))
    # model.add(Dropout(.3))
    model.add(ELU())
    
    model.add(Dense(50))
    model.add(ELU())
    
    model.add(Dense(10))
    model.add(ELU())
      
    model.add(Dense(1))
    adam = Adam(lr=learn_rate)
    model.compile(optimizer=adam, loss="mse") # since we're predicting continous driving angle, it make sense to use the mse loss.

    return model

def pre_img(img_batch,w,h):
    # Crop
    img_batch = list(map(lambda x: crop_img(x),img_batch))
    # Normalize
    # img_batch = list(map(lambda x: normalize(x),img_batch))
    # Resize
    return list(map(lambda x: cv2.resize(x,(w,h)),img_batch))

def normalize(img):
    a = -0.5
    b = 0.5
    xmin = 0
    xmax = 255
    return a+(img-xmin)*(b-a)/(xmax-xmin)
    
    
def crop_img(data):
    return data[32:160-25]

# function for visualize the prediction angle of images, for debugging
def visualize_train(ts,tl):
    ts = np.array([mpimg.imread(image_name) for image_name in ts[0:100]])
    tl = np.array(tl[0:100])
    ts,tl = shuffle(ts,tl)
    #randomly select 9 pictures to predict steering angle
    fig, axes = plt.subplots(3, 3, figsize=(8, 5),
                             subplot_kw={'xticks': [], 'yticks': []})
    
    fig.subplots_adjust(hspace=0.04, wspace=0.05)
    for i,ax in enumerate(axes.flat):
        if i< 9:
            sampleimage = ts[i]
            ax.imshow(sampleimage)
            ax.axis('off')
            ax.set_title(str(tl[i]))
        else:
            fig.delaxes(ax)
            
# helper function to find certain value in a list and return all indexes
def find_all_index(arr,item):
    return [i for i,a in enumerate(arr) if a==item]


if __name__ == "__main__":
    ### Read in augmented plus original images
    trains = pd.read_csv('t_csv.csv')
    vals = pd.read_csv('v_csv.csv')
    ts,tl = trains.ts,trains.tl
    vs,vl = vals.vs,vals.vl
    
    batch_size = 128
    # visualization
#    visualize_train(ts,tl)
#    plt.show()
    # fetch model
    
    # test to see which epoch works the best
#    for i in range(10):
#        model = create_model(learn_rate = 0.0001) #Nvidia's model
#        model.fit_generator(gen_batch_img(ts,tl,batch_size),
#                            samples_per_epoch=len(tl),nb_epoch=8+i,
#                            validation_data=gen_batch_img(vs,vl,batch_size),
#                            nb_val_samples=len(vl))
#        #print(model.summary())
#        print("Saving model!",i)
#        namesave = "model.h5"
#        model.save(str(i)+namesave)
     
    model = create_model(learn_rate = 0.0001) #Nvidia's model
    model.fit_generator(gen_batch_img(ts,tl,batch_size),
                        samples_per_epoch=len(tl),nb_epoch=10,
                        validation_data=gen_batch_img(vs,vl,batch_size),
                        nb_val_samples=len(vl))
    #print(model.summary())
    print("Saving model!")
    namesave = "model2.h5"
    model.save(namesave)
    
    
    
