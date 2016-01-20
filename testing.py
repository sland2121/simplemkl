"""
testing script for working_algo1_al.py (simpleMKL alg implementation)

partition on input:
1. kernel_functions
-# kernel functions: 1, 2, 3+
-types of kernel functions: gaussians, poly (not necessary because the type
of kernel does not affect simplemkl algorithm)
-case: equivalent kernels, different kernels

2. weight initializations (kinit):
-uniform
-non-uniform
	-0 weight on some kernels

3. C: 1, 10, 100, 1000

4. stopping criterion: duality gap, weights not changing, KKT



test cases:
1. 1 kernel (gaussian(5)). output weight=1.0
2. 4 kernels:
        -4 equivalent (no learning occurs):
		-kinit=uniform,non-uniform.
		-C: 1,10,100,1000
	-4 different kernels (learning occurs)
		-kinit: uniform, non-uniform:[1 0 0], [0 1 0], [0 0 1]
		-C:
                    1,1000: kinit matters
                    10,100: kinit does not matter
                    
		
	
"""
import numpy as np
import working_algo1_al as algo1
import working_kernel_helpers as k_helpers


def construct_gram_cube(xtrain,kernel_functions):
    # constructs a 3-D Gram matrix
    kernel_matrices=[]
    num_train=xtrain.shape[0]
    for m,kernel_func in enumerate(kernel_functions):
        kernel_matrices.append(np.empty((num_train,num_train)))
        #Creates kernel matrix
        for i in range(num_train):
            for j in range(num_train):
                kernel_matrices[m][i, j] = kernel_func(xtrain[i], xtrain[j])
        #unit trace normalization is needed
        trace=np.trace(kernel_matrices[m])
        kernel_matrices[m]=kernel_matrices[m]*(1.0/trace)
    return kernel_matrices
    
    
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


# test case 1: 1 kernel
print 'case 1'
k_init=np.array([1.0])
kernel_functions=[k_helpers.create_rbf_kernel(5.0)]
kernel_matrices=construct_gram_cube(xtrain,kernel_functions)
C=10
weights,final_K,J,alpha,duality_gap=algo1.find_kernel_weights(k_init,kernel_matrices,C,ytrain)
np.testing.assert_array_almost_equal(weights,np.array([1.0]))
#assert np.array_equal(weights,np.array([1.0]))

# test case 2: equivalent kernels (no learning)
# across different C

#case a: uniform weights

k_init=np.array([0.25,0.25,0.25,0.25])
kernel_functions=[k_helpers.create_rbf_kernel(5.0), k_helpers.create_rbf_kernel(5.0),
                  k_helpers.create_rbf_kernel(5.0),k_helpers.create_rbf_kernel(5.0)]
kernel_matrices=construct_gram_cube(xtrain,kernel_functions)
C=10
print 'case 2a'
weights,final_K,J,alpha,duality_gap=algo1.find_kernel_weights(k_init,kernel_matrices,C,ytrain)
#assert np.array_equal(weights,np.array([0.25,0.25,0.25,0.25]))
np.testing.assert_array_almost_equal(weights,np.array([0.25,0.25,0.25,0.25]))

print 'case 2b'
k_init=np.array([0.4,0.3,0.2,0.1])
weights,final_K,J,alpha,duality_gap=algo1.find_kernel_weights(k_init,kernel_matrices,C,ytrain)
np.testing.assert_array_almost_equal(weights,np.array([0.4,0.3,0.2,0.1]))

C=1
print 'case 2c'
k_init=np.array([0.25,0.25,0.25,0.25])
weights,final_K,J,alpha,duality_gap=algo1.find_kernel_weights(k_init,kernel_matrices,C,ytrain)
np.testing.assert_array_almost_equal(weights,np.array([0.25,0.25,0.25,0.25]))
print 'case 2d'
k_init=np.array([0.4,0.3,0.2,0.1])
weights,final_K,J,alpha,duality_gap=algo1.find_kernel_weights(k_init,kernel_matrices,C,ytrain)
np.testing.assert_array_almost_equal(weights,np.array([0.4,0.3,0.2,0.1]))

# test case 3: different kernels (learning)
# 3 kernels
kernel_functions=[k_helpers.create_rbf_kernel(1.0),k_helpers.create_rbf_kernel(3.0),
                  k_helpers.create_rbf_kernel(5.0)]
kernel_matrices=construct_gram_cube(xtrain,kernel_functions)
C=10
print 'case 3a'
k_init=np.array([1.0/3, 1.0/3, 1.0/3])
weights,final_K,J,alpha,duality_gap=algo1.find_kernel_weights(k_init,kernel_matrices,C,ytrain)
np.testing.assert_array_almost_equal(weights,np.array([0,1,0]))
print 'case 3b'
k_init=np.array([0,1,0])
weights,final_K,J,alpha,duality_gap=algo1.find_kernel_weights(k_init,kernel_matrices,C,ytrain)
np.testing.assert_array_almost_equal(weights,np.array([0,1,0]))

# NOTE: below test cases were manually checked against weights
# due to precision issues
# test case 4: different kernels (learning) across different C

kernel_functions=[k_helpers.create_rbf_kernel(1.0),k_helpers.create_rbf_kernel(3.0),
                  k_helpers.create_rbf_kernel(5.0),k_helpers.create_rbf_kernel(10.0)]
kernel_matrices=construct_gram_cube(xtrain,kernel_functions)

# case 4c-d: k_init does not influence the weights
C=10
#case a: uniform weights
print 'case 4a'
k_init=np.array([0.25,0.25,0.25,0.25])
weights,final_K,J,alpha,duality_gap=algo1.find_kernel_weights(k_init,kernel_matrices,C,ytrain)
np.testing.assert_array_almost_equal(weights,np.array([0,1,0,0]))

#remaining cases: non-uniform weights
print 'case 4b'
k_init=np.array([0,0,0,1])
weights,final_K,J,alpha,duality_gap=algo1.find_kernel_weights(k_init,kernel_matrices,C,ytrain)
np.testing.assert_array_almost_equal(weights,np.array([0,1,0,0]))

print 'case 4c'
k_init=np.array([1,0,0,0])
weights,final_K,J,alpha,duality_gap=algo1.find_kernel_weights(k_init,kernel_matrices,C,ytrain)
np.testing.assert_array_almost_equal(weights,np.array([0,1,0,0]))

# testing interaction of k_init and C values

# case 4c-1: k_init influences the weights in higher sigfigs
print "C=1"
C=1
k_init=np.array([1,0,0,0])
weights,final_K,J,alpha,duality_gap=algo1.find_kernel_weights(k_init,kernel_matrices,C,ytrain)
print weights
#np.testing.assert_array_almost_equal(weights,np.array([0,0.5159,0.4247,0.0594]))
k_init=np.array([0.25,0.25,0.25,0.25])
weights,final_K,J,alpha,duality_gap=algo1.find_kernel_weights(k_init,kernel_matrices,C,ytrain)
print weights
k_init=np.array([0,1,0,0])
weights,final_K,J,alpha,duality_gap=algo1.find_kernel_weights(k_init,kernel_matrices,C,ytrain)
print weights


# case 4c-2: k_init does not influence the weights
print "C=100"
C=100
k_init=np.array([0,1,0,0])
weights,final_K,J,alpha,duality_gap=algo1.find_kernel_weights(k_init,kernel_matrices,C,ytrain)
print weights
k_init=np.array([1,0,0,0])
weights,final_K,J,alpha,duality_gap=algo1.find_kernel_weights(k_init,kernel_matrices,C,ytrain)
print weights
k_init=np.array([0.25,0.25,0.25,0.25])
weights,final_K,J,alpha,duality_gap=algo1.find_kernel_weights(k_init,kernel_matrices,C,ytrain)
print weights
#np.testing.assert_array_almost_equal(weights,np.array([0,1,0,0]))

# case 4c-3: k_init influences weights in lower sigfigs
print "C=1000"
C=1000
k_init=np.array([0,1,0,0])
weights,final_K,J,alpha,duality_gap=algo1.find_kernel_weights(k_init,kernel_matrices,C,ytrain)
print weights
k_init=np.array([1,0,0,0])
weights,final_K,J,alpha,duality_gap=algo1.find_kernel_weights(k_init,kernel_matrices,C,ytrain)
print weights
k_init=np.array([0.25,0.25,0.25,0.25])
weights,final_K,J,alpha,duality_gap=algo1.find_kernel_weights(k_init,kernel_matrices,C,ytrain)
print weights
#np.testing.assert_array_almost_equal(weights,np.array([0.62210,0.3779,0,0]))

print 'All test cases passed.'
