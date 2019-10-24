import numpy as np 
kernel_sizes = []
f = open("kernel_sizes.txt","r")
file = f.read().split('\n')

try:
    for row in file:
        kernel_size = [0,0,0,0,0]
        row = row.split(",")
        for i in range(5):
            kernel_size[i] = int(row[i])
    
        kernel_sizes.append(kernel_size)
except:
    print("Hi i m exception :)")
    print(kernel_sizes)

