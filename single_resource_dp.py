# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 11:44:57 2018

@author: hyu
"""
import numpy as np

def single_resource_dp(num_product, num_steps, cap, product_prob, adj_revenue):
    # num_product scalar 
    # num_step scalar for now
    # cap is scalar
    # product_prob is num_product x 1
    # adj_revenue is num_product x 1
    
    #temporary inputs
    num_product = 28
    #num_steps = 128900
    num_steps = 1289
    cap = 100
    product_prob = np.asarray([9.9000e-01, 6.0436e-05, 5.9443e-04, 6.6212e-04, 1.1665e-04,
       7.8424e-05, 2.1006e-04, 2.3389e-05, 6.4966e-04, 4.6338e-04,
       7.2199e-04, 3.8423e-04, 2.9192e-04, 1.8903e-04, 4.6666e-04,
       2.5565e-04, 3.5800e-04, 5.6212e-04, 6.2947e-04, 7.6461e-04,
       1.7601e-04, 4.4020e-04, 3.0932e-05, 7.2659e-04, 2.1034e-04,
       4.4593e-04, 2.8384e-04, 2.0397e-04])
    adj_revenue = np.asarray([  0.    , 815.064 , 767.905 , 286.3545, 193.0943, 978.9122,
       406.2287, 757.7679,  89.1518, 309.8835, 618.9276, 459.9105,
       218.3094, 663.5202, 678.6827, 950.3185, 281.2611, 619.8517,
       383.2951, 400.3607, 942.6527, 929.9179, 948.3751, 375.4859,
       342.2961, 664.7771,  42.3206, 232.2399])
    
    # according to RM literature, tstep = 0 means no time left    
    
    #boundary conditions
    V_tm1 = np.full(cap+1, 0.0)
    for tstep in range(1, num_steps):
        V_t = np.full(cap+1, 0.0)        
        #no need to calculate V(0,t) since it is already 0.0
        for x in range(1, cap+1):
            rev = [max(V_tm1[x-1] + adj_revenue[p], V_tm1[x]) 
                    for p in range(0, num_product)]
            V_t[x] = np.dot(product_prob, rev)
        V_tm1 = V_t
                
        # V(x,t) = E[max (price(p)u + V(x-u,t-1))]
        # V(s,t) = max
    
    