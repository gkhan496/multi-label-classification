import scipy.io
import numpy as np
import os

mat = scipy.io.loadmat('left_data.mat')
X = []
y = []

#Convert .mat file to .csv files for Machine Learning in Python

"""for i in range(mat['train_data'].shape[0]):
    for j in range(14):
        f = open('eeg_dataset.txt',"a+") # or eeg_dataset.csv
        f.write(str(mat['train_data'][i][j]))
        f.write(',')

    f.write(str(mat['train_label'][i][0]))
    f.write('\n')


for i in range(mat['test_data'].shape[0]):
    for j in range(14):
        f = open('eeg_dataset.txt',"a+") # or eeg_dataset.csv
        f.write(str(mat['test_data'][i][j]))
        f.write(',')

    f.write(str(mat['test_label'][i][0]))
    f.write('\n')"""



file = open('information.txt','a+')
file.write(classes[i]+"-->"+get_bin(i)+"-->"+str(i))
file.write('\n')
##################################################################################
label = mat['data'].item()[0]
photos = mat['data'].item()[1]



for i in range(label.shape[0]):
    src = str(photos.item(i)[0])
    dst = str(label.item(i))+"-"+str(photos.item(i)[0])
    os.rename(src, dst)

"""for photo in enumerate(mat['data'].item()[1]):
    print(photo[0])"""