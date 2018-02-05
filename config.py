# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 23:02:12 2018

@author: hyu
"""

import numpy as np
import sys
import time

#save all parameters in one location

def param_init(param):
    
    #control all parameter settings
    case = 4
    
    param["case"] = case
    
    if case == 1:    
        seed_training = 4321          
        np.random.seed(seed_training)
        param["seed_training"] = seed_training
        
        seed_simulation = 12345
        param["seed_simulation"] = seed_simulation
        
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
        # adding product null we get 2N products
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
        
        #roughly speaking the price is a few hundred dollars
        product_revenue = 1000*np.random.uniform(size=[num_product])
        product_revenue[product_null] = 0
        param["product_revenue"] = product_revenue
        #total demand
        product_demand = np.random.uniform(size=[num_product])*capacity
        product_demand[product_null]  = 0
        param["product_demand"] = product_demand
    
        #total arrival rate should be less than or equal to 0.01
        #thus we get num of steps
        num_steps = int(np.sum(product_demand)/0.01)
        param["num_steps"] = num_steps
        
        #arrival rate (including product null)
        product_prob = np.divide(product_demand,num_steps)
        product_prob[0] = 1.0 - np.sum(product_prob)
        param["product_prob"] = product_prob
        
    if case == 2:    
        seed_training = 15485669          
        np.random.seed(seed_training)
        param["seed_training"] = seed_training
        
        seed_simulation = 15485773
        param["seed_simulation"] = seed_simulation
        
        #business parameter initialization
        num_nights = 14
        param["num_nights"] = num_nights
        capacity = 100
        param["capacity"] = capacity
        # product zero is the no-revenue no resource product
        # added for simplicity
        product_null = 0
        param["product_null"] = product_null
        # if there are N nights, adding product null gives us 3N-2 products
        # product 0 is null product 
        # product 1~N are 1-night
        # product N+1~2N-1 are 2-nighter
        # product 2N ~3N-3 are 3-nighter
        num_product = num_nights*3-2
        param["num_product"] = num_product
        product_resource_map = np.zeros((num_product, num_nights))
        for i in range(1, num_nights+1):    
            product_resource_map[i][i-1] = 1.0        
        for i in range(num_nights+1, 2*num_nights):            
            product_resource_map[i][i-num_nights-1] = 1.0
            product_resource_map[i][i-num_nights] = 1.0
        for i in range(2*num_nights, 3*num_nights-2):
            product_resource_map[i][i-2*num_nights] = 1.0
            product_resource_map[i][i-2*num_nights+1] = 1.0
            product_resource_map[i][i-2*num_nights+2] = 1.0
   
        
        #product_resource_map[num_product-1][num_nights-1] = 1.0    
        param["product_resource_map"] = product_resource_map
        
        #roughly speaking the price is a few hundred dollars
        product_revenue = 1000*np.random.uniform(size=[num_product])
        product_revenue[product_null] = 0
        param["product_revenue"] = product_revenue
        #total demand
        product_demand = np.random.uniform(size=[num_product])*capacity
        product_demand[product_null]  = 0
        param["product_demand"] = product_demand
    
        #total arrival rate should be less than or equal to 0.01
        #thus we get num of steps
        num_steps = int(np.sum(product_demand)/0.01)
        param["num_steps"] = num_steps
        
        #arrival rate (including product null)
        product_prob = np.divide(product_demand,num_steps)
        product_prob[0] = 1.0 - np.sum(product_prob)
        param["product_prob"] = product_prob        
        
    if case == 3:    
        seed_training = 15485567          
        np.random.seed(seed_training)
        param["seed_training"] = seed_training
        
        seed_simulation = 15485651
        param["seed_simulation"] = seed_simulation
        
        #business parameter initialization
        num_nights = 21
        param["num_nights"] = num_nights
        capacity = 1000
        param["capacity"] = capacity
        # product zero is the no-revenue no resource product
        # added for simplicity
        product_null = 0
        param["product_null"] = product_null
        # if there are N nights, adding product null gives us 3N-2 products
        # product 0 is null product 
        # product 1~N are 1-night
        # product N+1~2N-1 are 2-nighter
        # product 2N ~3N-3 are 3-nighter
        num_product = num_nights*3-2
        param["num_product"] = num_product
        product_resource_map = np.zeros((num_product, num_nights))
        for i in range(1, num_nights+1):    
            product_resource_map[i][i-1] = 1.0        
        for i in range(num_nights+1, 2*num_nights):            
            product_resource_map[i][i-num_nights-1] = 1.0
            product_resource_map[i][i-num_nights] = 1.0
        for i in range(2*num_nights, 3*num_nights-2):
            product_resource_map[i][i-2*num_nights] = 1.0
            product_resource_map[i][i-2*num_nights+1] = 1.0
            product_resource_map[i][i-2*num_nights+2] = 1.0
   
    if case == 4:    
        seed_training = 15485567          
        np.random.seed(seed_training)
        param["seed_training"] = seed_training
        
        seed_simulation = 15485651
        param["seed_simulation"] = seed_simulation
        
        #business parameter initialization
        num_nights = 14
        param["num_nights"] = num_nights
        capacity = 500
        param["capacity"] = capacity
        # product zero is the no-revenue no resource product
        # added for simplicity
        product_null = 0
        param["product_null"] = product_null
        # if there are N nights, adding product null gives us 3N-2 products
        # product 0 is null product 
        # product 1~N are 1-night
        # product N+1~2N-1 are 2-nighter
        # product 2N ~3N-3 are 3-nighter
        num_product = num_nights*3-2
        param["num_product"] = num_product
        product_resource_map = np.zeros((num_product, num_nights))
        for i in range(1, num_nights+1):    
            product_resource_map[i][i-1] = 1.0        
        for i in range(num_nights+1, 2*num_nights):            
            product_resource_map[i][i-num_nights-1] = 1.0
            product_resource_map[i][i-num_nights] = 1.0
        for i in range(2*num_nights, 3*num_nights-2):
            product_resource_map[i][i-2*num_nights] = 1.0
            product_resource_map[i][i-2*num_nights+1] = 1.0
            product_resource_map[i][i-2*num_nights+2] = 1.0        
        #product_resource_map[num_product-1][num_nights-1] = 1.0    
        param["product_resource_map"] = product_resource_map
        
        #roughly speaking the price is a few hundred dollars
        product_revenue = 1000*np.random.uniform(size=[num_product])
        product_revenue[product_null] = 0
        param["product_revenue"] = product_revenue
        #total demand
        product_demand = np.random.uniform(size=[num_product])*capacity
        product_demand[product_null]  = 0
        param["product_demand"] = product_demand
    
        #total arrival rate should be less than or equal to 0.01
        #thus we get num of steps
        num_steps = int(np.max(product_demand)/0.01)
        param["num_steps"] = num_steps
        
        #arrival rate (including product null)
        product_prob = np.divide(product_demand,num_steps)
        product_prob[0] = 1.0 - np.sum(product_prob)
        param["product_prob"] = product_prob                
    
    if case == 5:    
        seed_training = 15485357          
        np.random.seed(seed_training)
        param["seed_training"] = seed_training
        
        seed_simulation = 15485411
        param["seed_simulation"] = seed_simulation
        
        #business parameter initialization
        num_nights = 21
        param["num_nights"] = num_nights
        capacity = 500
        param["capacity"] = capacity
        # product zero is the no-revenue no resource product
        # added for simplicity
        product_null = 0
        param["product_null"] = product_null
        # if there are N nights, adding product null gives us 3N-2 products
        # product 0 is null product 
        # product 1~N are 1-night
        # product N+1~2N-1 are 2-nighter
        # product 2N ~3N-3 are 3-nighter
        num_product = num_nights*3-2
        param["num_product"] = num_product
        product_resource_map = np.zeros((num_product, num_nights))
        for i in range(1, num_nights+1):    
            product_resource_map[i][i-1] = 1.0        
        for i in range(num_nights+1, 2*num_nights):            
            product_resource_map[i][i-num_nights-1] = 1.0
            product_resource_map[i][i-num_nights] = 1.0
        for i in range(2*num_nights, 3*num_nights-2):
            product_resource_map[i][i-2*num_nights] = 1.0
            product_resource_map[i][i-2*num_nights+1] = 1.0
            product_resource_map[i][i-2*num_nights+2] = 1.0        
        #product_resource_map[num_product-1][num_nights-1] = 1.0    
        param["product_resource_map"] = product_resource_map
        
        #roughly speaking the price is a few hundred dollars
        product_revenue = 1000*np.random.uniform(size=[num_product])
        product_revenue[product_null] = 0
        param["product_revenue"] = product_revenue
        #total demand
        product_demand = np.random.uniform(size=[num_product])*capacity
        product_demand[product_null]  = 0
        param["product_demand"] = product_demand
    
        #total arrival rate should be less than or equal to 0.01
        #thus we get num of steps
        num_steps = int(np.max(product_demand)/0.01)
        param["num_steps"] = num_steps
        
        #arrival rate (including product null)
        product_prob = np.divide(product_demand,num_steps)
        product_prob[0] = 1.0 - np.sum(product_prob)
        param["product_prob"] = product_prob                        
        

    #all output files should have a timepstamp regardless
    # this way we can always sort out different version
    timestamp = time.strftime('%b-%d-%Y_%H_%M_%S', time.gmtime()).lower()
    param["timestamp"] = timestamp        
    
    if sys.platform == "win32":
        model_path = "C:/Users/hyu/Desktop/bellman/model/"
    elif sys.platform == "linux":
        model_path = "/home/ubuntu/model/"
    else:
        model_path = ""
        
    fname_output_model = "".join([model_path, "dpdnn.",timestamp, ".ckpt"])
    param["fname_output_model"] = fname_output_model
    
    #used to read policy file
    fname_policy = "".join([model_path, "lpdp_value_function.npy"])
    param["fname_policy"] = fname_policy        
    
    #used to WRITE policy file by lpdp.py
    fname_policy_output = "".join([model_path, "lpdp_value_function.", timestamp, ".npy"])
    param["fname_policy_output"] = fname_policy_output    
    
    #tensorflow model
    fname_tensorflow_model = "".join([model_path, "dpdnn.feb-01-2018_16_45_40.ckpt"])
    param["fname_tensorflow_model"] = fname_tensorflow_model
    
    fname_json = "".join([model_path, "dpdnn.", timestamp, ".json"])
    param["fname_json"] = fname_json    
    
    fname_npz = "".join([model_path, "batch_b64.npz"])
    param["fname_npz"] = fname_npz
    
    fname_validation = "".join([model_path, "simulation.", timestamp, ".npz"])
    param["fname_validation"] = fname_validation    
    
    debug_lp = 1
    param["debug_lp"] = debug_lp

    
    #computational graph parameters
    
    #define a state (in batch) and a linear value function
    batch_size = 64
    param["batch_size"] = batch_size
    
    #the state of the MDP is fully represented by number of rooms available
    # each night plus time remaining
    # but this is subject to change
    # if we decide to add dual value or utilization metrics to state
    # this is the place
    dim_state_space = num_nights+1
    param["dim_state_space"] = dim_state_space
    
    #hidden layer size
    hidden = 256
    param["hidden"] = hidden
    
    num_hidden_layer = 5
    param["num_hidden_layer"] = num_hidden_layer
    
    #weights initialzation parameters
    init_level = 1.0
    param["init_level"] = init_level
    init_level_output = 1.0
    param["init_level_output"] = init_level_output
    
    num_batches_training = 1000
    param["num_batches_training"] = num_batches_training            
    
    save_param(param)

def save_param(param):
    import json    
    tmp = dict(zip(param.keys(), param.values()))
    tmp["product_resource_map"] = tmp["product_resource_map"].tolist()
    tmp["product_revenue"] = tmp["product_revenue"].tolist()
    tmp["product_demand"] = tmp["product_demand"].tolist()
    tmp["product_prob"] = tmp["product_prob"].tolist()
    with open(tmp["fname_json"], 'w') as fp:
        json.dump(tmp, fp)    

#testing
if 0:
    conf = dict()
    param_init(conf)    
