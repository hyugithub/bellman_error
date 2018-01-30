# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 10:10:18 2018

@author: hyu
"""

import config  
import numpy as np
from lp_module import lp 
from single_resource_dp import single_resource_dp
import time

def lpdp(param):    
    #do a lp and use dual to do dp for each night
    param = conf
    num_nights = param["num_nights"]
    num_product = param["num_product"]
    capacity = param["capacity"]
    product_demand = param["product_demand"]
    product_revenue = param["product_revenue"]
    product_resource_map = param["product_resource_map"]
    _, dual = lp(np.full(num_nights, capacity)
                , product_demand
                , conf
                , True)
    print("dual = ", dual)
    
    # for every product p and night, first of all
    # adj_revenue = 0 if p does not use night
    # secondly if 
    adj_revenue = np.reshape(np.repeat(product_revenue, num_nights), (-1, num_nights))    
    adj_revenue = np.multiply(adj_revenue, product_resource_map)

    #clumsy loop but shold not take much time
    for p in range(num_product):
        for night in range(num_nights):
            if product_resource_map[p][night] <= 1e-6:
                #product p and night are independnet
                continue                
            # weneed to perform adjustment
            for n2 in range(num_nights):
                if product_resource_map[p][n2] >=1.0 and n2 != night:
                    adj_revenue[p][night] -= dual[n2]
    
    #adjustment for every product, every night
    
    result = np.stack([single_resource_dp(adj_revenue[:,night]
                                        , product_resource_map[:,night]
                                        , conf)
                        for night in range(num_nights)])
    fname_policy_output = param["fname_policy_output"] 
    np.save(fname_policy_output, result)
#    for night in range(num_nights):
#        print("solve DP for night", night)
#        adj_rev = adj_revenue[:,night]
#        resource = product_resource_map[:,night]
#        single_resource_dp(adj_rev, resource, conf)
    return 1.0

ts = time.time()
np.set_printoptions(precision=4)
conf = dict()
config.param_init(conf)
lpdp(conf)

print("total number to be saved: "
      , conf["num_steps"]*conf["num_nights"]*conf["capacity"]
      )
print("total policy generation time = %.1f" % (time.time()-ts))