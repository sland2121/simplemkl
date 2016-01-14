"""
Author: Sayeri Lala
Date: 1-14-16

Code implementing simpleMKL algorithm: http://www.jmlr.org/papers/volume9/rakotomamonjy08a/rakotomamonjy08a.pdf,
as implemented in matlab version: http://asi.insa-rouen.fr/enseignants/~arakoto/code/mklindex.html.
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
"""
def find_kernel_weights(k_init,kernel_matrices,C,y):
    
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
    alpha, J= helpers.compute_J_SVM(combined_kernel_matrix, y_mat,C)
    
    while(not stop_state):
        #print "iteration:",iteration
        #print "d:",d
        

        dJ = helpers.compute_dJ(kernel_matrices, y_mat, alpha)
        #print 'gradient before entering while loop'
        #print dJ
        
        mu=np.argmax(d)
        D = helpers.compute_descent_direction(d, dJ,mu)

        gamma_max=helpers.compute_max_admissible_gamma(d,D)
        # done in matlab version
        if gamma_max>0.1:
            gamma_max=0.1
            
        J_cross=0
        J_prev=J
        while (J_cross < J):
            
            d_cross = d + gamma_max*D
            combined_kernel_matrix_cross = k_helpers.get_combined_kernel(kernel_matrices, d_cross)
            alpha_cross, J_cross= helpers.compute_J_SVM(combined_kernel_matrix_cross, y_mat,C)
            
            if J_cross<J:
                J=J_cross
                d = d_cross.copy()
                alpha=alpha_cross.copy()
                # update descent
                D=helpers.update_descent_direction(d,D,mu)
                
                # gamma_max=helpers.compute_max_admissible_gamma(d,D)
                
                tmp_ind=np.where(D<0)[0]      
                if tmp_ind.shape[0]>0:
                    gamma_max=np.min(-(np.divide(d[tmp_ind],D[tmp_ind])))
                    J_cross=0
                else:
                    gamma_max=0
        
        # line-search             
        gamma, alpha, J=helpers.compute_gamma_linesearch(0,gamma_max,J,J_cross,
                                               d,D,kernel_matrices, J_prev,y_mat,alpha)
        #print 'descent',D                                      
        #print 'gamma after line search',gamma
        d = d + gamma * D
        #print 'weights after linesearch',d
        
        # compute duality gap for terminating condition
#        combined_kernel_matrix = k_helpers.get_combined_kernel(kernel_matrices, d)
#        alpha_curr_d, J_curr_d = helpers.compute_J_SVM(combined_kernel_matrix, y_mat,C,-90)

#        print 'support vectors used to compute current gradient'
#        print alpha
        dJ_curr_d = helpers.compute_dJ(kernel_matrices, y_mat, alpha)

        # another formulation of the duality gap provided in matlab
        
##        print J
##        print np.max(-dJ_curr_d)
##        print np.sum(alpha)
##        print 'numerator', (J+np.max(-dJ_curr_d) -np.sum(alpha))
        
        duality_gap=(J+np.max(-dJ_curr_d) -np.sum(alpha))/J
        #print 'duality gap: ',duality_gap
        if duality_gap<0.01:
            stop_state=True
        """
        duality_gap=J_curr_d+0.5*alpha_curr_d.dot(np.multiply(combined_kernel_matrix,y_mat)).dot(alpha_curr_d.T)\
                    -np.sum(alpha_curr_d)
        
        """
        
        iteration += 1
    return (d,k_helpers.get_combined_kernel(kernel_matrices, d),J,alpha,duality_gap)
