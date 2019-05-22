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

    def __init__(self,H_L,train_path,test_path,result_name,number_of_neurons,batch_size,lr,dropout,epoch,class_num):

        self.H_L = H_L
        self.train_path = train_path
        self.test_path = test_path
        self.img_w = 227
        self.img_h = 227
        self.result_name = result_name
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
        baslangic = input("Kaçtan başlaşın : ")
        self.kernel_sizes = self.kernel_sizes[int(baslangic):]
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


    def train(self,tr_img_data,tr_lbl_data,tst_img_data,tst_lbl_data,slice,prediction=0):
        
        for kernel_size in self.kernel_sizes:
            try:
                model = Sequential()
                model.add(InputLayer(input_shape=[self.img_w,self.img_h,1]))

                model.add(Conv2D(filters=32,kernel_size=kernel_size[0],strides=1,activation='relu'))
                model.add(MaxPool2D(pool_size=2))

                model.add(Conv2D(filters=64,kernel_size=kernel_size[1],strides=1,activation='relu'))
                model.add(MaxPool2D(pool_size=2))

                model.add(Conv2D(filters=128,kernel_size=kernel_size[2],strides=1,activation='relu'))
                model.add(MaxPool2D(pool_size=2))

                model.add(Conv2D(filters=256,kernel_size=kernel_size[3],strides=1,activation='relu'))
                model.add(MaxPool2D(pool_size=2))

                model.add(Flatten())
                
                for i in range(len(self.H_L)):
                    model.add(Dense(int(self.H_L[i]),activation='relu'))
                                
                            
                model.add(Dense(self.class_num,activation='softmax'))
                Optimizer = Adam(lr=0.70e-5)
                model.compile(optimizer=Optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
                model.fit(x=tr_img_data,y=tr_lbl_data,batch_size=self.batch_size, epochs=10, verbose=1)
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
    parser.add_argument('--hidden_layers', type=int, default=3)
    parser.add_argument('--result_name', type=str, default='results.txt')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.70e-5)
    parser.add_argument('--n_of_neurons', type=int, default=121)
    parser.add_argument('--dropout',type=float,default=0.2)
    parser.add_argument('--class_num',type=int,default=2,help="Number of class")
    parser.add_argument('-H_L','--H_L', action='append', required=True)


    args = parser.parse_args()


from labeling import Label_Processing as lp
rename = input("Yeniden isimlendirmek istiyor musunuz ? Y/N")
if rename == "Y":
    lp().rename_for_label()
parameters = input("Parametreleri değiştirmek istiyor musunuz ? Y/N")
if rename == "N":
    print("as")

train = train_CNN(hidden_layers=args.H_L,train_path="train",test_path="test",result_name=args.result_name,batch_size=16,
lr=args.learning_rate,epoch=args.epoch,dropout=args.dropout,class_num=args.class_num)
train.train_data()

# python train.py --hiddensize=3 --epoch=200 --learning_rate=0.001 --class_num=4 --dropout=0.3 --n_of_neurons=121