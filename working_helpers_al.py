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

weight_threshold=1e-08

def fix_weight_precision(d,weight_precision):
    new_d=d.copy()
    # zero out weights below threshold
    new_d[np.where(d<weight_precision)[0]]=0
    # normalize
    new_d=new_d/np.sum(new_d)
    return new_d
        
def compute_J_SVM(K,y_mat,C):
    clf=svm.SVC(C=C,kernel='precomputed')    
    clf.fit(K,y_mat[:,0])
    #signed_alpha=clf.dual_coef_[0]*-1
    alpha=clf.dual_coef_[0]
    # dual coefficients should be nonnegative
    for i in np.where(alpha<0)[0]:
        alpha[i]*=-1
    alpha_indices=clf.support_
    #return alpha_indices,alpha
##    print "# svs",alpha_indices.shape[0]
    #print "sv indices: ",np.sort(alpha_indices)
    
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
##    print "not svs",np.setdiff1d()
    J= -0.5*(np.absolute(complete_alpha).dot(np.multiply(K,y_mat)).dot(np.absolute(complete_alpha.T)))\
       +np.sum(np.absolute(complete_alpha))   
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

def update_descent_direction(d,D,mu,weight_precision):
    tmp_ind=np.intersect1d(np.where(D<=0)[0],np.where(d<=weight_precision)[0])
    D[tmp_ind]=0
    
    if mu == 0:
        D[mu]=-np.sum(D[mu+1:])
    else:
        D[mu] = -np.sum(np.concatenate((D[0:mu],D[mu+1:]),0))   
    return D
    
def compute_max_admissible_gamma(d,D):
     #max admissible step size
    tmp_ind=np.where(D<0)[0]

    if tmp_ind.shape[0]>0:
        gamma_max=np.min(-(np.divide(d[tmp_ind],D[tmp_ind])))
    else:
        gamma_max=0
        
    return gamma_max

def compute_gamma_linesearch(gamma_min,gamma_max,delta_max,cost_min,cost_max,d,D,kernel_matrices,J_prev,y_mat,
                             alpha,C,goldensearch_precision_factor):
    gold_ratio=(5**0.5+1)/2

##    print "stepmin",gamma_min
##    print "stepmax",gamma_max
##    print "deltamax",delta_max
    gamma_arr=np.array([gamma_min,gamma_max])
    cost_arr=np.array([cost_min,cost_max])

    coord=np.argmin(cost_arr)
##    print 'linesearch conditions'
##    print 'gamma_min',gamma_min
##    print 'gamma_max',gamma_max
##    print 'delta_max',delta_max
##    print 'golden search precision factor', goldensearch_precision_factor
    
    while ((gamma_max-gamma_min) > goldensearch_precision_factor*(abs(delta_max)) and gamma_max>np.finfo(float).eps):
        # print 'in line search loop'
        gamma_medr=gamma_min+(gamma_max-gamma_min)/gold_ratio;
        gamma_medl=gamma_min+(gamma_medr-gamma_min)/gold_ratio;

        tmp_d = d + gamma_medr*D
        alpha_r,cost_medr=compute_J_SVM(k_helpers.get_combined_kernel(kernel_matrices,tmp_d),y_mat,C)
        tmp_d=d+gamma_medl*D
        alpha_l,cost_medl=compute_J_SVM(k_helpers.get_combined_kernel(kernel_matrices,tmp_d),y_mat,C)
    
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
        dJ[m] = -0.5 * alpha.dot(np.multiply(kernel_matrix,y_mat)).dot(alpha.T)
        #dJ[m] = -0.5 * alpha.dot(kernel_matrix).dot(alpha.T)

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


