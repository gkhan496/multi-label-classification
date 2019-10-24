import argparse
import time
import cv2
import numpy as np 
import os
from random import shuffle
from tqdm import tqdm
import keras
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import TensorBoard
from sklearn.metrics import confusion_matrix 
import seaborn as sns


class train_CNN():

    def __init__(self,hidden_size,train_path,test_path,result_name,number_of_neurons,batch_size,lr,dropout,epoch,class_num):

        self.hidden_size = hidden_size
        self.train_path = train_path
        self.test_path = test_path
        self.img_w = 200
        self.img_h = 250
        self.result_name = result_name
        self.number_of_neurons = number_of_neurons
        self.batch_size = batch_size
        self.class_num = class_num
        self.lr = []
        self.epoch = []
        self.dropout = []
        self.kernel_sizes = []
        self.slice = slice
        
        super().__init__()

    def train_data(self):
        file = open(self.result_name,"a+")
        file.write("Model-Accuracy-Sensitivity-Specificity-F1Score-TPR-FPR-Hidden_Size-Number_Of_Neurons-Epoch-Learning_Rate-Batch_Size")
        file.write("\n")
        file.close()
        training_images = self.train_data_with_label(self.train_path)
        testing_images = self.test_data_with_label(self.test_path)
        tr_img_data = np.array([i[0] for i in training_images]).reshape(-1,self.img_w,self.img_h,1) 
        tr_lbl_data = np.array([i[1] for i in training_images])
        tst_img_data = np.array([i[0] for i in testing_images]).reshape(-1,self.img_w,self.img_h,1)
        tst_lbl_data = np.array([i[1] for i in testing_images])
        self.kernel_sizes = []
        f = open("kernel_sizes.txt","r")
        file = f.read().split('\n')

        try:
            for row in file:
                kernel_size = [0,0,0,0,0]
                row = row.split(",")
                for i in range(5):
                    kernel_size[i] = int(row[i])
            
                self.kernel_sizes.append(kernel_size)
        except:
            print("")

        self.train(tr_img_data,tr_lbl_data,tst_img_data,tst_lbl_data,self.slice)

    def one_hot_label(self,img):
        ohl = []
        label = img.split('-')[0]
        for i in range(self.class_num):
            ohl.append(0)
        ohl[int(label)] = 1
        return ohl
        
    def train_data_with_label(self,train_data):
        train_images = []
        for i in tqdm(os.listdir(train_data)):
            
            path = os.path.join(train_data,i)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img,(self.img_w,self.img_h))
            train_images.append([np.array(img), self.one_hot_label(i)])
        shuffle(train_images)
        return train_images

    def test_data_with_label(self,test_data):
        test_images = []
        for i in tqdm(os.listdir(test_data)):
            path = os.path.join(test_data,i)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img,(self.img_w,self.img_h))
            test_images.append([np.array(img), self.one_hot_label(i)])
        shuffle(test_images)
        return test_images


    def train(self,tr_img_data,tr_lbl_data,tst_img_data,tst_lbl_data,slice,prediction=0,):
        #print(self.batch_size,self.epoch,self.hidden_size,self.number_of_neurons)
        print(len(self.lr))
        for kernel_size in self.kernel_sizes:
            try:       
                for p in range(5): 

                    for t in range(5):                    

                        model = Sequential()
                        model.add(InputLayer(input_shape=[self.img_w,self.img_h,1]))

                        model.add(Conv2D(filters=32,kernel_size=kernel_size[0],strides=1,activation='relu'))
                        model.add(MaxPool2D(pool_size=2))

                        model.add(Conv2D(filters=64,kernel_size=kernel_size[1],strides=1,activation='relu'))
                        model.add(MaxPool2D(pool_size=2))

                        model.add(Flatten())
                        model.add(Dense(121,activation='relu'))
                        model.add(Dense(121,activation='relu'))
                        model.add(Dense(self.class_num,activation='softmax'))
                        Optimizer = Adam(lr=0.80e-5)
                        model.compile(optimizer=Optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
                        model.fit(x=tr_img_data,y=tr_lbl_data,batch_size=self.batch_size, epochs=60, verbose=1)
                        model.summary()


                        acc = 0
                        prediction = model.evaluate(tst_img_data,tst_lbl_data)
                        print(prediction)
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

            except Exception as e:
                print("Heyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy")
                print(e)
                continue

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Auto MultiLabel Classification With CNN')
    parser.add_argument('--hiddensize', type=int, default=3)
    parser.add_argument('--result_name', type=str, default='results.txt')
    parser.add_argument('--batch_size', type=int, default=16)
    #parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.70e-5)
    parser.add_argument('--n_of_neurons', type=int, default=121)


    args = parser.parse_args()

from labeling import Label_Processing as lp

train = train_CNN(hidden_size=args.hiddensize,train_path="train",test_path="test",result_name=args.result_name,batch_size=16,number_of_neurons=args.n_of_neurons,
lr=[0.90e-5,0.001,1e-5],epoch=[1,2,3],dropout=[0.4,0.5],class_num=6)
train.train_data()