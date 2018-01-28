# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 23:02:12 2018

@author: hyu
"""

import numpy as np
import sys

#save all parameters in one location

def param_init(param):
    seed_training = 4321
    np.random.seed(seed_training)
    param["seed_training"] = seed_training
    
    seed_simulation = 12345
    param["seed_simulation"] = seed_simulation
    
    if sys.platform == "win32":
        model_path = "C:/Users/hyu/Desktop/bellman/model/"
    elif sys.platform == "linux":
        model_path = "/home/ubuntu/model/"
    else:
        model_path = ""
        
    fname_output_model = model_path+"dp.ckpt"
    
    debug_lp = 1
    param["debug_lp"] = debug_lp
    
    #business parameter initialization
    num_nights = 14
    param["num_nights"] = num_nights
    capacity = 100
    param["capacity"] = capacity
    # product zero is the no-revenue no resource product
    # added for simplicity
    product_null = 0
    param["product_null"] = product_null
    # unfortunately, to avoid confusion we need to add a fairly 
    # complex product matrix
    # if there are N nights, there are N one-night product from 
    # 1 to N; there are also N-1 two-night products from N+1 to 2N-1
    num_product = num_nights*2
    param["num_product"] = num_product
    product_resource_map = np.zeros((num_product, num_nights))
    for i in range(1,num_nights):
        product_resource_map[i][i-1] = 1.0
        product_resource_map[i][i] = 1.0
    for i in range(0,num_nights):    
        product_resource_map[i+num_nights][i] = 1.0
    #product_resource_map[num_product-1][num_nights-1] = 1.0    
    param["product_resource_map"] = product_resource_map
    
    product_revenue = 1000*np.random.uniform(size=[num_product])
    product_revenue[product_null] = 0
    param["product_revenue"] = product_revenue
    #total demand
    product_demand = np.random.uniform(size=[num_product])*capacity
    product_demand[product_null]  = 0
    param["product_demand"] = product_demand
    
    num_steps = int(np.sum(product_demand)/0.01)
    param["num_steps"] = num_steps
    
    #arrival rate (including)
    product_prob = np.divide(product_demand,num_steps)
    product_prob[0] = 1.0 - np.sum(product_prob)
    param["product_prob"] = product_prob
    
    #computational graph generation
    
    #define a state (in batch) and a linear value function
    batch_size = 16
    param["batch_size"] = batch_size
    #LHS is the value function for current state at time t
    #for each state, we need num_nights real value inputs for available
    # inventory, and +1 for time
    dim_state_space = num_nights+1
    param["dim_state_space"] = dim_state_space
    
    #tensorflow model inputs (or really state space samples)
    #V(s,t)
    #try neural network model: input->hidden->output
    
    num_batches_training = 11
    param["num_batches_training"] = num_batches_training
    
    #policy_list = ["fifo", "dnn"]
    #param["policy_list"] = policy_list

if 0:
    conf = dict()
    param_init(conf)