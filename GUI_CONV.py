import sys 
from PyQt5 import QtWidgets,QtGui


class TrainCNN(QtWidgets.QWidget):
    
    def __init__(self, parent=None, flags=Qt.WindowFlags()):
        super().__init__()
        self.


app = QtWidgets.QApplication(sys.argv)

win = QtWidgets.QWidget()

win.setWindowTitle("Classification Tool")


vertical = QtWidgets.QVBoxLayout()
horiz = QtWidgets.QHBoxLayout()

b_train = QtWidgets.QPushButton("TRAIN")

horiz.addStretch()
horiz.addWidget(b_train)

vertical.addStretch()
vertical.addLayout(horiz)

win.setLayout(vertical)



win.setGeometry(100,100,600,600)
win.show()
sys.exit(app.exec_())

import time
import cv2
import numpy as np 
import scipy.stats as st
import os
from random import shuffle
from tqdm import tqdm
import keras
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import TensorBoard
#import matplotlib.pyplot as plt
import keras.backend as K


class train_CNN():

    def __init__(self, activation='softmax',*args, **kwargs):

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epoch = epoch
        self.number_of_epoch = number_of_epoch
        
        self.activation = activation

        return super().__init__(*args, **kwargs)




    def train(self,tr_img_data,tr_lbl_data,tst_img_data,tst_lbl_data,img_w,img_h):
        f = open("kernel_sizes.txt","r")
        kernel_sizes = f.read().split('\n')
        kernel_size = [0,0,0,0,0]
        prediction = 0
        for row in kernel_sizes:
            row = row.split(",")
            for i in range(5):
                kernel_size[i] = int(row[i])    
            try:
                for p in range(5):
                    
                    model = Sequential()
                    
                    model.add(InputLayer(input_shape=[img_w,img_h,1]))

                    model.add(Conv2D(filters=32,kernel_size=kernel_size[0],strides=1,activation='relu'))
                    model.add(MaxPool2D(pool_size=2))

                    model.add(Conv2D(filters=64,kernel_size=kernel_size[1],strides=1,activation='relu'))
                    model.add(MaxPool2D(pool_size=2))

                    model.add(Conv2D(filters=128,kernel_size=kernel_size[2],strides=1,activation='relu'))
                    model.add(MaxPool2D(pool_size=2))

                    model.add(Conv2D(filters=256,kernel_size=kernel_size[3],strides=1,activation='relu'))
                    model.add(MaxPool2D(pool_size=2))
                    model.add(Flatten())



                    model.add(Dropout(0.2))
                    model.add(Dense(121,activation='relu'))


                    model.add(Dense(self.class_num,activation=self.activation))
                    Optimizer = Adam(lr=self.learning_rate)

                    #model.compile(optimizer=Optimizer,loss='categorical_crossentropy',metrics=['accuracy']) #Parametrelerine bak
                    #Cross entropy grafiği MAE MSE
                    #checkpoint = keras.callbacks.ModelCheckpoint('model-{epoch:03d}.h5', verbose=1, monitor='val_acc',save_best_only=True, mode='auto')  

                    model.compile(optimizer=Optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

                    model.fit(x=tr_img_data,y=tr_lbl_data,batch_size=10, epochs=1, verbose=1)
                    model.summary()

                    acc = 0
                    prediction = model.evaluate(tst_img_data,tst_lbl_data)
                    if prediction[1] > acc : 
                        acc = prediction[1]
                        strr = str(kernel_size[0])+"-"+str(kernel_size[1])+"-"+str(kernel_size[2])+"-"+str(kernel_size[3])
                        f = open("Accuracies.txt","a+")
                        f.write(strr+":"+"BEST____ACC : "+str(prediction[1]))
                        f.write('\n')
                        model.save_weights("bestacc.h5")
                        f.close()
                    else:
                        strr = str(kernel_size[0])+"-"+str(kernel_size[1])+"-"+str(kernel_size[2])+"-"+str(kernel_size[3])
                        f = open("Accuracies.txt","a+")
                        f.write(strr+":"+"ACC : "+str(prediction[1]))
                        f.write('\n')

                        f.close()
                    K.clear_session()
                    #kernel_size.clear()
            except Exception as e:
                print(e)
                continue
        return prediction


    

    def train_data_with_label(train_data,img_w,img_h,class_num):
        try:
            train_images = []
            for i in tqdm(os.listdir(train_data)):
                path = os.path.join(train_data,i)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img,(img_w,img_h))
                train_images.append([np.array(img), one_hot_label(i,class_num)])
            shuffle(train_images)
        except:
            print(path)
        return train_images

    def test_data_with_label(test_data,img_w,img_h,class_num):
        test_images = []
        for i in tqdm(os.listdir(test_data)):
            path = os.path.join(test_data,i)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img,(img_w,img_h))
            test_images.append([np.array(img), one_hot_label(i,class_num)])
        shuffle(test_images)
        return test_images
    classes = []
    for i in os.listdir("train"):
        classes.append(i)

    #class_num = input("Enter the class number : ")
    class_num = 27
    train_data = 'train/'
    img_w,img_h = 227,227
    training_images = train_data_with_label(train_data,img_w,img_h,class_num)
    test_data = 'test/' 
    testing_images = test_data_with_label(test_data,img_w,img_h,class_num)
    tr_img_data = np.array([i[0] for i in training_images]).reshape(-1,img_w,img_h,1) 
    tr_lbl_data = np.array([i[1] for i in training_images])
    tst_img_data = np.array([i[0] for i in testing_images]).reshape(-1,img_w,img_h,1)
    tst_lbl_data = np.array([i[1] for i in testing_images])

    print(train(tr_img_data,tr_lbl_data,tst_img_data,tst_lbl_data,img_w,img_h,class_num))