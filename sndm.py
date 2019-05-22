"""import numpy as np 
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

"""
"""import argparse
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
print(len(args.H_L))

"""
f = open("kernel_sizes.txt","r")
file = f.read().split('\n')
kernel_sizes = []
try:
    for row in file:
        kernel_size = [0,0,0,0,0]
        row = row.split(",")
        for i in range(5):
            kernel_size[i] = int(row[i])
            
        kernel_sizes.append(kernel_size)
except:
    print("")

kernel_sizes = kernel_sizes[:]
print(kernel_sizes[0])