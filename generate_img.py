# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 22:15:23 2017

@author: xubug34
"""

## for generating additional data
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
import os
from sklearn.utils import shuffle

# function for augment new data based on flag given
def read_store(sample_names,labels,flag):
    
    names = []
    angles = []
    # flip half of the samples randomly
    sample_names,labels = shuffle(sample_names,labels)
    half = int(len(sample_names)/2)    
    # create new image based on user's option
    for s,l in zip(sample_names[0:half],labels[0:half]):
        img = mpimg.imread(s)
        if flag == "shift":
            new_img,new_label = shift_center(img,l,100)
            # store imgs
            new_name = ".\IMG\shift_"+os.path.split(s)[1]
        elif flag == "flip":
            new_label = -l
            new_img = np.fliplr(img)
            new_name = ".\IMG\\flip_"+os.path.split(s)[1]
        else:
            raise AssertionError()
        mpimg.imsave(new_name,new_img)
        # update the lists to be returned
        names.append(new_name)
        angles.append(new_label)
        
    sample_names = sample_names+names
    labels = labels+angles
    print("Generate complete")
    return sample_names, labels

# function that reads in left and right images
def triple_img(data):
    train_left = [i.strip() for i in data.left]
    train_right = [i.strip() for i in data.right]
    train_center = list(data.center)
    
    train = train_center+train_left+train_right    
    # for recovering from left or right edge
    label_center = np.array(data.angle)
    label_center = label_center*1.2 # added for testing steeper turns
    # a test function to test for penatlty for driving off road
#    maxl = abs(label_center).max()
#    minl = 0
#    f = lambda x: 0.3*(abs(x)-minl)/(maxl-minl)  
    
    label_center = list(label_center) # convert back to list
  
    label_left = [i+0.3 for i in data.angle]
    label_right = [i-0.3 for i in data.angle]
    label = label_center+label_left+label_right
 
    return train,label

# augment shifted images, not used
def shift_center(img,angle,shift_range):
    x_dir = np.random.uniform(-shift_range/2,shift_range/2)
    new_angle = angle+x_dir/shift_range*2*.2 #?? shoulde be angle+x_dir*.2*.2
    #new_angle = angle+x_dir*.2*.2
    y_dir = np.random.uniform(-15,15)
    Trans = np.float32([[1,0,x_dir],[0,1,y_dir]])
    new_img = cv2.warpAffine(img,Trans,(320,160))
#    plt.imshow(new_img)
#    print(new_angle)
    return new_img,new_angle

# main function to generate and store images    
def generate_train_val():
    print("start generating new images")
    file_name = 'driving_log.csv'
    df = pd.read_csv(file_name)
    df = shuffle(df)
    df.columns = ['center','left','right','angle','throttle','break','speed']
    
    # Pre remove zero    
    dfnozero = df[(df['angle']!=0)]
    dfwithzero = df[(df['angle']==0)]
    dfnew = dfnozero.append(dfwithzero[0:int(0.2*len(dfwithzero))])
    # Split the data
    train,val = train_test_split(dfnew,test_size=0.2,random_state=0)
    # Trip the data, this include adding left and right camera data
        
    ts,tl = triple_img(train)
    vs,vl = triple_img(val)

#    ts,tl = remove_zero(ts,tl,0.9)
#    vs,vl = remove_zero(vs,vl,0.9)
    
    ts,tl = read_store(ts,tl,"flip")    
    vs,vl = read_store(vs,vl,"flip")

    ts,tl = shuffle(ts,tl)
    vs,vl = shuffle(vs,vl)
    
    print("Generate complete --ALL")
    return ts,tl,vs,vl

# function that removes unessary zeros, not used. 
# implemented another approach in gen_train_val
def remove_zero(samples,labels,k_percent):
    count_index = [i for i,a in enumerate(labels) if abs(a) == 0]
    #randomly select 3/4
    zeros_len = len(count_index)
    print("The labels is of size:",len(labels))
    print("The numbers of zeros in the labels are:",zeros_len)
    # calculate the numbers to ditch
    to_ditch = int((1-k_percent)*zeros_len)
    print("Caculated ditch number is:",to_ditch)
    # randomly select their postions
    
    #to_ditch_indexs = [np.random.randint(0,len(count_index)) for i in range(to_ditch)]#bug
    count_index = shuffle(count_index)
    to_ditch_indexs = count_index[0:to_ditch]
    
    samples,labels = remove_element(samples,labels,to_ditch_indexs)
    print("The labels now have total of:",len(labels))
    return samples,labels

# helper function to remove zeros
def remove_element(list1,list2,index):
    clip1=[]
    clip2=[]
    for i in range(len(list1)):
        if i not in index:
            clip1.append(list1[i])
            clip2.append(list2[i])
            
    return clip1,clip2
    

if __name__ == "__main__":
    
    ts,tl,vs,vl = generate_train_val()
    t_raw = {'ts':ts,'tl':tl}
    v_raw = {'vs':vs,'vl':vl}

    t_df = pd.DataFrame(t_raw)
    v_df = pd.DataFrame(v_raw)
    
    t_df.to_csv('t_csv.csv')
    v_df.to_csv('v_csv.csv')
    print("Saving CSV complete")

    
    