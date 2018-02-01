# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 11:24:06 2018

@author: hyu
"""

#the purpose of this file is to run NN training only

import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import time
#from ortools.linear_solver import pywraplp
import itertools
from lp_module import lp
from simulation import *

import config
from functional_approximator import *

#general initialization
ts = time.time()
ops.reset_default_graph()
np.set_printoptions(precision=4)

#all parameters
conf = dict()
config.param_init(conf)

seed_training = conf["seed_training"]
np.random.seed(seed_training)

seed_simulation = conf["seed_simulation"]  
fname_output_model = conf["fname_output_model"]
debug_lp = conf["debug_lp"]

#business parameter initialization
num_nights = conf["num_nights"] 
capacity = conf["capacity"]
# product zero is the no-revenue no resource product
# added for simplicity
product_null = conf["product_null"]
# unfortunately, to avoid confusion we need to add a fairly 
# complex product matrix
# if there are N nights, there are N one-night product from 
# 1 to N; there are also N-1 two-night products from N+1 to 2N-1
num_product = conf["num_product"]
product_resource_map = conf["product_resource_map"] 
product_revenue = conf["product_revenue"] 

product_demand = conf["product_demand"] 

num_steps = conf["num_steps"]

#arrival rate (including)
product_prob = conf["product_prob"] 

#computational graph generation

#define a state (in batch) and a linear value function
batch_size = conf["batch_size"] 
#LHS is the value function for current state at time t
#for each state, we need num_nights real value inputs for available
# inventory, and +1 for time
dim_state_space = conf["dim_state_space"] 

#tensorflow model inputs (or really state space samples)
#V(s,t)
#try neural network model: input->hidden->output

#define linear approximation model
#model = error_model_linear()
model = error_model_simple_nn(conf)
model.build()
conf["model"] = model

num_batches_training = conf["num_batches_training"] 
#sg = sample_generation(conf)
sg = sample_generation_prebulit(conf)
first_run = True

saver = tf.train.Saver()
   
with tf.Session() as sess:    
    conf["sess"] = sess
    sess.run(tf.global_variables_initializer())    

    for batch in range(num_batches_training):
        #generate data for LHS V(s,t)
        
        #data_lhs, data_rhs_1, data_rhs_2, data_mask, lp_bound_lhs, lp_bound_rhs_1, lp_bound_rhs_2 = generate_batch(conf, sg)
        data_lhs, data_rhs_1, data_rhs_2, data_mask, lp_bound_lhs, lp_bound_rhs_1, lp_bound_rhs_2 = generate_batch_from_file(conf, sg)
        
        if first_run:
            first_run = False
            print("Before even training, check parameters:")
            model.read_loss(sess
                    , data_lhs
                    , data_rhs_1
                    , data_rhs_2
                    , data_mask
                    , lp_bound_lhs
                    , lp_bound_rhs_1
                    , lp_bound_rhs_2)
            model.read_param(sess
                    , data_lhs
                    , data_rhs_1
                    , data_rhs_2
                    , data_mask
                    , lp_bound_lhs
                    , lp_bound_rhs_1
                    , lp_bound_rhs_2)
            model.read_gradients(sess
                    , data_lhs
                    , data_rhs_1
                    , data_rhs_2
                    , data_mask
                    , lp_bound_lhs
                    , lp_bound_rhs_1
                    , lp_bound_rhs_2)                        
            print("Before even training, check parameters end\n\n")
                
        #we will have to run the session twice since tensorflow does
        # not ensure all tasks are executed in a pre-determined order per
        # https://github.com/tensorflow/tensorflow/issues/13133
        # this is the training step
        model.train(sess
                    , data_lhs
                    , data_rhs_1
                    , data_rhs_2
                    , data_mask
                    , lp_bound_lhs
                    , lp_bound_rhs_1
                    , lp_bound_rhs_2)
        # statistics accumulation
        if 1 and batch % 100 == 0:              
            print("batch = ", batch, " validation")
            print("on training:")
            print("check %.4f"%np.sum(lp_bound_lhs))
            model.read_loss(sess
                , data_lhs
                , data_rhs_1
                , data_rhs_2
                , data_mask
                , lp_bound_lhs
                , lp_bound_rhs_1
                , lp_bound_rhs_2)
            if 1:
                #get a different batch and sample again
                #data_lhs, data_rhs_1, data_rhs_2, data_mask, lp_bound_lhs, lp_bound_rhs_1, lp_bound_rhs_2 = generate_batch(conf, sg)                
                data_lhs, data_rhs_1, data_rhs_2, data_mask, lp_bound_lhs, lp_bound_rhs_1, lp_bound_rhs_2 = generate_batch_from_file(conf, sg)
                print("on new validation")
                print("check %.4f"%np.sum(lp_bound_lhs))
                model.read_loss(sess
                    , data_lhs
                    , data_rhs_1
                    , data_rhs_2
                    , data_mask
                    , lp_bound_lhs
                    , lp_bound_rhs_1
                    , lp_bound_rhs_2)
#            model.read_gradients(sess
#                    , data_lhs
#                    , data_rhs_1
#                    , data_rhs_2
#                    , data_mask
#                    , lp_bound_lhs
#                    , lp_bound_rhs_1
#                    , lp_bound_rhs_2)            
            print("\n")
            sys.stdout.flush() 
            
    save_path = saver.save(sess, fname_output_model) 
    
    print("validation for random samples:")    
    for vb in range(1):
        print("validation batch ", vb)
        #data_lhs, data_rhs_1, data_rhs_2, data_mask, lp_bound_lhs, lp_bound_rhs_1, lp_bound_rhs_2 = generate_batch(conf, sg)
        data_lhs, data_rhs_1, data_rhs_2, data_mask, lp_bound_lhs, lp_bound_rhs_1, lp_bound_rhs_2 = generate_batch_from_file(conf, sg)
        model.read_loss(sess
                    , data_lhs
                    , data_rhs_1
                    , data_rhs_2
                    , data_mask
                    , lp_bound_lhs
                    , lp_bound_rhs_1
                    , lp_bound_rhs_2)
        
    print("validation for monotonicity when state = constant:")    
    for vb in range(1):
        print("validation batch ", vb)
        #data_lhs, data_rhs_1, data_rhs_2, data_mask, lp_bound_lhs, lp_bound_rhs_1, lp_bound_rhs_2 = generate_batch_t0()
        data_lhs, data_rhs_1, data_rhs_2, data_mask, lp_bound_lhs, lp_bound_rhs_1, lp_bound_rhs_2 = generate_batch_fix_time(conf)
#        model.read_loss(sess
#                    , data_lhs
#                    , data_rhs_1
#                    , data_rhs_2
#                    , data_mask
#                    , lp_bound_lhs
#                    , lp_bound_rhs_1
#                    , lp_bound_rhs_2)        
        val = model.predict(sess, data_lhs, lp_bound_lhs)
        print(val)

    print("total model building time = %.2f seconds" % (time.time()-ts), " time per batch = %.2f sec"%((time.time()-ts)/num_batches_training))
    
