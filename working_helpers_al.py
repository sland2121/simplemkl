"""
Author: Sayeri Lala
Date: 1-14-16

Helper functions used in working_algo1_al.py.

These helper functions were implemented based
on matlab implementation of simpleMKL: http://asi.insa-rouen.fr/enseignants/~arakoto/code/mklindex.html

compute_dJ, get_armijos_step_size were implemented by gjtrowbridge:
https://github.com/gjtrowbridge/simple-mkl-python/blob/master/helpers.py
"""

import numpy as np
import working_kernel_helpers as k_helpers
from sklearn import svm


def compute_J_SVM(K,y_mat,C):
    clf=svm.SVC(C=C,kernel='precomputed')    
    clf.fit(K,y_mat[:,0])
    alpha=clf.dual_coef_[0]
    # dual coefficients should be nonnegative
    for i in np.where(alpha<0)[0]:
        alpha[i]*=-1
    alpha_indices=clf.support_

    complete_alpha=np.zeros((y_mat[:,0].shape[0]))
    #possible support vector indices
    #create ordered list of support vector coefficients
    for i in range(complete_alpha.shape[0]):
        
        if i not in alpha_indices:
            continue #point is not a support vector
        else:
            tmp_index=np.where(alpha_indices == i)
            assert len(tmp_index)==1
            tmp_index=tmp_index[0][0]
            complete_alpha[i]=alpha[tmp_index]
   
    J= -0.5*(complete_alpha.dot(np.multiply(K,y_mat)).dot(complete_alpha.T))+np.sum(complete_alpha)   
    return complete_alpha,J


def compute_descent_direction(d, dJ,mu):
    # normalizing the gradient
    norm_grad=dJ.dot(dJ)

    grad_new=(dJ*1.0)/(norm_grad**0.5)
    grad_new = grad_new-grad_new[mu]
    
    tmp_ind=np.intersect1d(np.where(d<=0)[0],np.where(grad_new>=0)[0])
    D=-1*grad_new
    D[tmp_ind]=0
    D[mu]=-np.sum(D)

    return D

def update_descent_direction(d,D,mu):
    tmp_ind=np.intersect1d(np.where(D<=0)[0],np.where(d<=0)[0])
    D[tmp_ind]=0
    if mu == 0:
        D[mu]=-np.sum(D[mu+1:])
    else:
        D[mu] = -np.sum(np.concatenate((D[0:mu-1],D[mu+1:]),0))
        
    return D
    
def compute_max_admissible_gamma(d,D):
     #max admissible step size
    tmp_ind=np.where(D<0)[0]

    if tmp_ind.shape[0]>0:
        gamma_max=np.min(-(np.divide(d[tmp_ind],D[tmp_ind])))
    else:
        gamma_max=0
        
    return gamma_max

def compute_gamma_linesearch(gamma_min,gamma_max,cost_min,cost_max,d,D,kernel_matrices,J_prev,y_mat,alpha):
    gold_ratio=(5**0.5+1)/2
    
    delta_max=gamma_max
    gamma_arr=np.array([gamma_min,gamma_max])
    cost_arr=np.array([cost_min,cost_max])

    coord=np.argmin(cost_arr)
    
    while ((gamma_max-gamma_min) > 0.1*(abs(delta_max))):
        gamma_medr=gamma_min+(gamma_max-gamma_min)/gold_ratio;
        gamma_medl=gamma_min+(gamma_medr-gamma_min)/gold_ratio;

        tmp_d = d + gamma_medr*D
        alpha_r,cost_medr=compute_J_SVM(k_helpers.get_combined_kernel(kernel_matrices,tmp_d),y_mat,10)
        tmp_d=d+gamma_medl*D
        alpha_l,cost_medl=compute_J_SVM(k_helpers.get_combined_kernel(kernel_matrices,tmp_d),y_mat,10)
    
        cost_arr=np.array([cost_min, cost_medl, cost_medr, cost_max])
        gamma_arr=np.array([gamma_min, gamma_medl, gamma_medr, gamma_max])
       
        coord=np.argmin(cost_arr)

        if coord==0:
            gamma_max=gamma_medl
            cost_max=cost_medl
            alpha=alpha_l
        if coord==1:
            gamma_max=gamma_medr
            cost_max=cost_medr
            alpha=alpha_r
        if coord==2:
            gamma_min=gamma_medl
            cost_min=cost_medl
            alpha=alpha_l
        if coord==3:
            gamma_min=gamma_medr
            cost_min=cost_medr
            alpha=alpha_r
        
    if cost_arr[coord] < J_prev:
        return gamma_arr[coord], alpha,cost_arr[coord]
    else:
        return gamma_min,alpha,cost_min

# this function was implemented in: https://github.com/gjtrowbridge/simple-mkl-python/blob/master/helpers.py
def compute_dJ(kernel_matrices, y_mat, alpha,debug=False):
    M = len(kernel_matrices)
    dJ = np.zeros(M)

    for m in range(M):
        kernel_matrix = kernel_matrices[m]
        if debug and m==1:
            print 'dJ for 2nd kernel computation'
#            print kernel_matrix[4,0:4]
            print alpha.dot(np.multiply(kernel_matrix,y_mat)).dot(alpha.T)
#        print 'kernel matrix: ',m
#        print kernel_matrix[0:2,0:9]
        dJ[m] = -0.5 * alpha.dot(np.multiply(kernel_matrix,y_mat)).dot(alpha.T) #based on matlab implementation
        #dJ[m] = -0.5 * alpha.dot(np.multiply(kernel_matrix, y_mat)).dot(alpha.T)        
        # assert dJ[m].shape[0] == 1

    return dJ

# this function was implemented in : https://github.com/gjtrowbridge/simple-mkl-python/blob/master/helpers.py
def get_armijos_step_size(iteration,C,kernel_matrices, d, y_mat, alpha0,gamma0, Jd, D, dJ, c=0.5, T=0.5):
#    print 'descent direction in armijos function'
#    print D    
    
    #m = D' * dJ, should be negative
    #Loop until f(x + gamma * p <= f(x) + gamma*c*m)
    # J(d + gamma * D) <= J(d) + gamma * c * m
    gamma = gamma0
    m = D.T.dot(dJ)

    while True:
        combined_kernel_matrix = k_helpers.get_combined_kernel(kernel_matrices, d + gamma * D)

        alpha, new_J,alpha_indices= compute_J_SVM(combined_kernel_matrix, y_mat,C)

        if new_J <= Jd + gamma * c * m:
            return gamma
        else:
            #Update gamma
            gamma = gamma * T
    return gamma
    #return gamma / 2


