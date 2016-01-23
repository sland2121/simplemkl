"""
Author: Sayeri Lala
Date: 1-14-16

Python transcription of the Matlab code (http://asi.insa-rouen.fr/enseignants/~arakoto/code/mklindex.html)
implementing simpleMKL algorithm: http://www.jmlr.org/papers/volume9/rakotomamonjy08a/rakotomamonjy08a.pdf.

"""

"""
kernel_matrices: Stores all the individual kernel matrices
each kernel matrix row/column is ordered in same order of points as y
returns weights and combined weighted kernel
 
""" 

import numpy as np
import working_helpers_al as helpers
import working_kernel_helpers as k_helpers

"""
k_init: numpy array of size len(kernel_matrices) summing to 1
kernel_matrices: list of kernel matrices
C: regularization parameter in SVM
y: labels for train points

stopping criterion: duality gap
"""

def find_kernel_weights(k_init,kernel_matrices,C,y):

    # various parameters
    weight_precision=1e-08 #weights below this value are set to 0
    goldensearch_precision=1e-01
    goldensearch_precision_init=1e-01
    max_goldensearch_precision=1e-08
    duality_gap_threshold=0.01
    
    for m in kernel_matrices:
        assert m.shape == (y.shape[0],y.shape[0])

    M = len(kernel_matrices)

    #initial weights of each kernel
    d = k_init
    
    #Creates y matrix for use in SVM later
    y_mat = np.outer(y, y)

    iteration = 0
    # initialization for stopping criterion
    stop_state=False
    
    # initial alphas
    combined_kernel_matrix = k_helpers.get_combined_kernel(kernel_matrices, d)
    alpha,J=helpers.compute_J_SVM(combined_kernel_matrix, y_mat,C)
    
##    print 'initial alpha, J: ',alpha, J
    while(not stop_state):
##	print "iteration:",iteration
##	print "d:",d
        old_d=d.copy()
        dJ = helpers.compute_dJ(kernel_matrices, y_mat, alpha)
##        print 'gradient before entering while loop'
##        print dJ
        mu=np.argmax(d)
        D = helpers.compute_descent_direction(d, dJ,mu)
##        print 'initial descent: ',D
        gamma_max=helpers.compute_max_admissible_gamma(d,D)
        delta_max=gamma_max

        if gamma_max>0.1:
            gamma_max=0.1
            
        J_cross=0
        J_prev=J

        while (J_cross < J):
            
##            print 'cost min: ',J
##            print 'cost max: ',J_cross
            d_cross = d + gamma_max*D
            combined_kernel_matrix_cross = k_helpers.get_combined_kernel(kernel_matrices, d_cross)
            alpha_cross, J_cross= helpers.compute_J_SVM(combined_kernel_matrix_cross, y_mat,C)
##            print "updated cost max: ",J_cross

            if J_cross<J:
                J=J_cross
                d = d_cross.copy()
##                print 'updated weights: ', d
                alpha=alpha_cross.copy()
                # update descent
##                print 'descent before update: ',D
                D=helpers.update_descent_direction(d,D,mu,weight_precision)
##                print 'updated descent: ', D
                
                # gamma_max=helpers.compute_max_admissible_gamma(d,D)
##                
                tmp_ind=np.where(D<0)[0]      
                if tmp_ind.shape[0]>0:
                    gamma_max=np.min(-(np.divide(d[tmp_ind],D[tmp_ind])))
                    delta_max=gamma_max
                    J_cross=0
                else:
                    gamma_max=0
                    delta_max=0
        # print 'support vector before line search',-np.sum(abs(alpha))
        # line-search
        gamma, alpha, J=helpers.compute_gamma_linesearch(0,gamma_max,delta_max,J,J_cross,
                                               d,D,kernel_matrices, J_prev,y_mat,alpha,C,goldensearch_precision)
        # print 'support vector after line search',-np.sum(abs(alpha))
##        print 'weights before final update',d
##        print 'gamma after line search',gamma
##        print 'descent',D                      
        d = d + gamma * D
        # numerical cleaning
        d = helpers.fix_weight_precision(d,weight_precision)
##        print 'weights after final update: ',d

        # improve line search by enhancing precision
        if max(abs(d-old_d))<weight_precision and \
            goldensearch_precision>max_goldensearch_precision:
                goldensearch_precision=goldensearch_precision/10
##        elif goldensearch_precision!=goldensearch_precision_init:
##            goldensearch_precision*=10
        

        
##        print 'weights after linesearch',d
        
#        print 'support vectors used to compute current gradient'
#        print alpha
        dJ_curr_d = helpers.compute_dJ(kernel_matrices, y_mat, alpha)

##        print 'parameters in computing duality gap'
##        print J
##        print np.max(-dJ_curr_d)
##        print -np.sum(alpha)
        
        # stopping criterion
        duality_gap=(J+np.max(-dJ_curr_d) -np.sum(alpha))/J
        # print 'duality gap: ',duality_gap
        if duality_gap<duality_gap_threshold:
            stop_state=True

        iteration += 1

        
    return (d,k_helpers.get_combined_kernel(kernel_matrices, d),J,alpha,duality_gap)
