"""
Author: Sayeri Lala
Date: 1-14-16

Code testing simpleMKL implementation.
This code was adapted from gitrowbridge:
https://github.com/gjtrowbridge/simple-mkl-python/blob/master/run_simpleMKL_for_ionosphere.py.

simpleMKL implementation details:
simpleMKL algorithm: http://www.jmlr.org/papers/volume9/rakotomamonjy08a/rakotomamonjy08a.pdf
simpleMKL matlab version: http://asi.insa-rouen.fr/enseignants/~arakoto/code/mklindex.html

"""

import numpy as np
import working_algo1_al as algo1
import working_kernel_helpers as k_helpers


# pre-processing
data_file = 'ionosphere.data'

data = np.genfromtxt(data_file, delimiter=',', dtype='|S10')
data=data[:,2:]
x = np.array(data[:, :-1], dtype='float')
y = np.array(data[:, -1] == 'b', dtype='int')
y[np.where(y == 0)] = -1

xtrain = x[:200]  # first 200 examples
ytrain = y[:200]  # first 200 labels

xtest = x[200:]  # example 201 onwards
ytest = y[200:]  # label 201 onwards

# normalize data
xtrain_mean_std=(np.mean(xtrain,axis=0),np.std(xtrain,axis=0))
xtest_mean_std=(np.mean(xtest,axis=0),np.std(xtest,axis=0))

xtrain=(xtrain-np.ones(xtrain.shape)*np.mean(xtrain,axis=0))/np.std(xtrain,axis=0)
xtest=(xtest-np.ones(xtest.shape)*np.mean(xtest,axis=0))/np.std(xtest,axis=0)


#gamma = 1.0/d
intercept = 0


kernel_functions=[
    k_helpers.create_rbf_kernel(4.0),
    k_helpers.create_rbf_kernel(5.0),
    k_helpers.create_rbf_kernel(5.0)
]



kernel_matrices=[]
num_train=xtrain.shape[0]

# constructs a 3-D Gram matrix
for m,kernel_func in enumerate(kernel_functions):
    kernel_matrices.append(np.empty((num_train,num_train)))
    #Creates kernel matrix
    for i in range(num_train):
        for j in range(num_train):
            kernel_matrices[m][i, j] = kernel_func(xtrain[i], xtrain[j])
    #unit trace normalization is needed
    trace=np.trace(kernel_matrices[m])
    kernel_matrices[m]=kernel_matrices[m]*(1.0/trace) 
      

weights,final_K,J,alpha,duality_gap=algo1.find_kernel_weights(xtrain,kernel_matrices,ytrain,C=10)

print 'weights: ',weights
print 'objective value: ', J
print 'duality gap: ', duality_gap
