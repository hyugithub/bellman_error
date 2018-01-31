#the purpose of this file is to generate a simple functional
# approximator to V(s,t)

import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import time
#from ortools.linear_solver import pywraplp
import itertools
from lp_module import lp
from simulation import simulation
import config

class error_model_simple_nn:
    def __init__(self, param):
        #inputs
        
        self.batch_size = param["batch_size"]
        self.num_nights = param["num_nights"]
        self.dim_state_space = param["dim_state_space"]
        self.num_product = param["num_product"]
        self.debug_lp = param["debug_lp"]
        self.hidden = param["hidden"]
        self.product_revenue = param["product_revenue"]
        self.product_prob = param["product_prob"]
        num_hidden_layer = param["num_hidden_layer"]
        #roughly speaking, we need to initialize 
        self.init_level = param["init_level"]
        self.init_level_output = param["init_level_output"]
        
        
        self.state_lhs = tf.placeholder(tf.float32, [self.batch_size, self.dim_state_space])
        #V(s-a(p),t-1)
        self.state_rhs_1 = tf.placeholder(tf.float32, [self.batch_size, self.num_product, self.dim_state_space])
        #V(s,t-1)
        self.state_rhs_2 = tf.placeholder(tf.float32, [self.batch_size, self.dim_state_space])
        self.mask = tf.placeholder(tf.float32, [self.batch_size, self.num_product])                
        
        if self.debug_lp:
            self.lp_bound_lhs = tf.placeholder(tf.float32, [self.batch_size])
            self.lp_bound_rhs_1 = tf.placeholder(tf.float32, [self.batch_size, self.num_product])
            self.lp_bound_rhs_2 = tf.placeholder(tf.float32, [self.batch_size])       
               
        self.value_lhs = tf.constant(0.0, dtype=tf.float32)
        self.value_rhs_1 = tf.constant(0.0, dtype=tf.float32)
        self.value_rhs_2 = tf.constant(0.0, dtype=tf.float32)
        self.weight_input = tf.Variable(self.init_level*np.random.normal(size=[self.dim_state_space, self.hidden])
                                        , dtype=tf.float32
                                        , name="weight_input"
                                        )
        self.bias_input = tf.Variable(self.init_level*np.random.normal(size=[self.hidden])
                                        , dtype=tf.float32
                                        , name="bias_input"
                                        )
        
        #Note: if output layer is linear, we should initialize this layer
        # separately since the output is equivalent to revenue
        # given output from previous layer is between 0 and 1, essentially
        # all parameters in this layer should be initialized
        # correspondingly...
        # if we switch to LP-adjusted model, these weights 
        # should be initialized differently
        self.weight_output = tf.Variable(self.init_level_output*np.random.normal(size=[self.hidden,1])
                                        , dtype=tf.float32
                                        , name="weight_output"
                                        )
        self.bias_output = tf.Variable(self.init_level_output*np.random.normal(size=[1,1])
                                        , dtype=tf.float32
                                        , name="bias_output"
                                        )
        
        # we just realize that we no longer need boundary conditions
        self.lambda_s0 = 0.0
        self.lambda_t0 = 0.0
        
        

        # determine each layer
        # 0 -- sigmoid
        # default: linear
        self.flag = [0]*(num_hidden_layer+1)
        self.flag[-1] = 1
        
        #with hidden units we can initialize size
        hidden_units = [self.hidden] * num_hidden_layer        
        self.weight_hidden = [tf.Variable(self.init_level*np.random.normal(size=[din,dout])
                                        , dtype=tf.float32
                                        , name="weight_hidden"
                                        ) 
                    for din,dout in zip(hidden_units, hidden_units[1:])]
        self.bias_hidden = [tf.Variable(np.random.uniform(size=[dout])
                                        , dtype=tf.float32
                                        , name="bias_hidden") 
                    for din,dout in zip(hidden_units, hidden_units[1:])]

    def build(self):
#        self.loss = self.generate_bellman_error_deep() \
#                        + self.generate_boundary_error_deep()
        self.loss = self.generate_bellman_error_deep() \
                        + tf.multiply(tf.constant(0.0, dtype=tf.float32), self.generate_boundary_error_deep())
        self.train_step =tf.train.AdagradOptimizer(0.1).minimize(self.loss)
        self.gradients = tf.gradients(self.loss, tf.trainable_variables())

    # given a sequence of weights and biases, build network
    # use flag to control layer
    def build_network(self, state_input, weights, biases):
        state = state_input
        #print(len(weights), state.shape)
        for w,b,m in zip(weights, biases, self.flag):
            if m == 0:
                # sigmoid = 0
                state = tf.nn.sigmoid(tf.matmul(state, w) + b)
            else:
                # linear = 1, default
                state = tf.matmul(state, w) + b
            #print("print state", state)
        return state

    def generate_bellman_error_deep(self):       
        weights = [self.weight_input] + self.weight_hidden + [self.weight_output]
        biases = [self.bias_input] + self.bias_hidden + [self.bias_output]        
        #V(s,t)        
        V_s_t = self.build_network(self.state_lhs, weights, biases)
        if self.debug_lp:
            #print(V_s_t, self.lp_bound_lhs)
            V_s_t = tf.multiply(V_s_t, tf.reshape(self.lp_bound_lhs, [self.batch_size, -1]))
        #V(s,t-1) 
        V_s_tm1 = self.build_network(self.state_rhs_2, weights, biases)
        if self.debug_lp:
            V_s_tm1 = tf.multiply(V_s_tm1, tf.reshape(self.lp_bound_rhs_2, [self.batch_size, -1]))
        #V(s-a(p),t-1)
        V_s1_tm1 = self.build_network(tf.reshape(self.state_rhs_1,[-1, self.dim_state_space]), weights, biases)
        V_s1_tm1 = tf.reshape(V_s1_tm1, [self.batch_size,-1])   
        if self.debug_lp:            
            V_s1_tm1 = tf.multiply(V_s1_tm1, self.lp_bound_rhs_1)
#            print(V_s_t)
#            print(V_s_tm1)
#            print(V_s1_tm1)
        
        value_lhs = V_s_t
        self.value = value_lhs
        value_rhs_2 = V_s_tm1
        
        #V(s-a(p),t-1) - V(s,t-1) + r(p)
        value_rhs_1 = V_s1_tm1 - tf.reshape(V_s_tm1, [self.batch_size,-1]) + tf.constant(self.product_revenue, dtype=tf.float32)
        # max(,0)
        value_rhs_1 = tf.maximum(value_rhs_1, tf.constant(0.0))
        # mask unsellable product
        value_rhs_1 = tf.multiply(value_rhs_1, self.mask)
        # multiply by probability
        value_rhs_1 = tf.multiply(value_rhs_1
                                , tf.constant(self.product_prob
                                              , dtype=tf.float32))        
        # sum
        value_rhs_1 = tf.reshape(tf.reduce_sum(value_rhs_1, axis=1), [self.batch_size,-1])        
        value_rhs = value_rhs_1 + value_rhs_2        
        #print(value_rhs)
        
        bellman_error = value_lhs - value_rhs
        #print(bellman_error)
        self.bellman_error = tf.reduce_mean(tf.multiply(bellman_error,bellman_error))
        self.value_lhs = value_lhs 
        self.value_rhs_1 = value_rhs_1
        self.value_rhs_2 = value_rhs_2
        #print(self.bellman_error)
        return self.bellman_error                
    
    def generate_boundary_error_deep(self): 
        #V(s,0) = 0
        
        mask_t0 = tf.reshape(tf.constant(np.concatenate([np.ones(self.num_nights), np.zeros(1)]), dtype=tf.float32), [self.dim_state_space, -1])
        weights_t0 = tf.multiply(self.weight_input, mask_t0)                 
        weights = [weights_t0] + self.weight_hidden + [self.weight_output]
        biases = [self.bias_input] + self.bias_hidden + [self.bias_output]                
        #V(s,t=0)
        v_t0 = self.build_network(self.state_lhs, weights, biases)
        boundary_t0 = tf.multiply(tf.reduce_mean(tf.multiply(v_t0,v_t0)), tf.constant(self.lambda_t0, dtype=tf.float32))
        
        mask_s0 = tf.reshape(tf.constant(np.concatenate([np.zeros(self.num_nights), np.ones(1)]), dtype=tf.float32), [self.dim_state_space, -1])
        weights_s0 = tf.multiply(self.weight_input, mask_s0)
        weights = [weights_s0] + self.weight_hidden + [self.weight_output]        
        #V(s=0,t)
        v_s0 = self.build_network(self.state_lhs, weights, biases)                
        boundary_s0 = tf.multiply(tf.reduce_mean(tf.multiply(v_s0,v_s0)), tf.constant(self.lambda_s0, dtype=tf.float32))
        self.boundary_error = boundary_t0 + boundary_s0
        return self.boundary_error       

    def train(self
              , session
              , data_lhs
              , data_rhs_1
              , data_rhs_2
              , data_mask
              , lp_bound_lhs = None
              , lp_bound_rhs_1 = None
              , lp_bound_rhs_2 = None
              ):
        if self.debug_lp:
            session.run(self.train_step
                    , feed_dict={self.state_lhs: data_lhs
                                 , self.state_rhs_1: data_rhs_1
                                 , self.state_rhs_2: data_rhs_2
                                 , self.mask: data_mask
                                 , self.lp_bound_lhs: lp_bound_lhs
                                 , self.lp_bound_rhs_1: lp_bound_rhs_1
                                 , self.lp_bound_rhs_2: lp_bound_rhs_2
                                 })
        else:
            session.run(self.train_step
                    , feed_dict={self.state_lhs: data_lhs
                                 , self.state_rhs_1: data_rhs_1
                                 , self.state_rhs_2: data_rhs_2
                                 , self.mask: data_mask
                                 })        

    def predict(self
                  , session
                  , data_lhs 
                  , lp_bound_lhs
                  ):
        return session.run(self.value_lhs
                , feed_dict={self.state_lhs: data_lhs
                             , self.lp_bound_lhs: lp_bound_lhs})       
    
    def read_loss(self
                  , session
                  , data_lhs 
                  , data_rhs_1 
                  , data_rhs_2 
                  , data_mask
                  , lp_bound_lhs = None
                  , lp_bound_rhs_1 = None
                  , lp_bound_rhs_2 = None                  
                  ):
        if self.debug_lp:
            result_loss, result_bellman, result_boundary, result_value = session.run(
                [self.loss, self.bellman_error, self.boundary_error, self.value_lhs]
                , feed_dict={self.state_lhs: data_lhs
                             , self.state_rhs_1: data_rhs_1
                             , self.state_rhs_2: data_rhs_2
                             , self.mask: data_mask
                             , self.lp_bound_lhs: lp_bound_lhs
                             , self.lp_bound_rhs_1: lp_bound_rhs_1
                             , self.lp_bound_rhs_2: lp_bound_rhs_2                             
                             })            
        else:
            result_loss, result_bellman, result_boundary, result_value = session.run(
                [self.loss, self.bellman_error, self.boundary_error, self.value_lhs]
                , feed_dict={self.state_lhs: data_lhs
                             , self.state_rhs_1: data_rhs_1
                             , self.state_rhs_2: data_rhs_2
                             , self.mask: data_mask
                             })            
        print("loss = %.3f"%result_loss, "bellman error = %.3f"%result_bellman, "boundary error = %.3f"%result_boundary)

    def read_gradients(self
                       , session
                       , data_lhs
                       , data_rhs_1
                       , data_rhs_2
                       , data_mask
                       , lp_bound_lhs = None
                       , lp_bound_rhs_1 = None
                       , lp_bound_rhs_2 = None                  
                      ):
        #read we have set up the network and it is running
        #all we need to do is to refer to objects we created
        # and are interested
        param = tf.trainable_variables()
        if self.debug_lp:
            result = session.run(
                [self.gradients] + param
                , feed_dict={self.state_lhs: data_lhs
                             , self.state_rhs_1: data_rhs_1
                             , self.state_rhs_2: data_rhs_2
                             , self.mask: data_mask
                             , self.lp_bound_lhs: lp_bound_lhs
                             , self.lp_bound_rhs_1: lp_bound_rhs_1
                             , self.lp_bound_rhs_2: lp_bound_rhs_2
                             })            
        else:
            result = session.run(
                [self.gradients] + param
                , feed_dict={self.state_lhs: data_lhs
                             , self.state_rhs_1: data_rhs_1
                             , self.state_rhs_2: data_rhs_2
                             , self.mask: data_mask
                             })
        gradients = result[0]
        values = result[1:]
        print("check gradients:")    
        for g,p,v in zip(gradients, param, values):
            print("parameter "
                  , p.name           
                  , "mean value = %.3f"%np.mean(v)
                  , "gradient mean=%.3f"%np.mean(g)
                  , "absmean=%.3f"%np.mean(np.absolute(g)))
        print("check gradients end")    
        
    def read_param(self
                   , session
                   , data_lhs
                   , data_rhs_1
                   , data_rhs_2
                   , data_mask
                   , lp_bound_lhs = None
                   , lp_bound_rhs_1 = None
                   , lp_bound_rhs_2 = None
                  ):
        #read we have set up the network and it is running
        #all we need to do is to refer to objects we created
        # and are interested
        if self.debug_lp:
            w_input, b_input, w_output, b_output, w_hidden, b_hidden = session.run(
                [self.weight_input, self.bias_input, self.weight_output, self.bias_output, self.weight_hidden, self.bias_hidden]
                , feed_dict={self.state_lhs: data_lhs
                             , self.state_rhs_1: data_rhs_1
                             , self.state_rhs_2: data_rhs_2
                             , self.mask: data_mask
                             , self.lp_bound_lhs: lp_bound_lhs
                             , self.lp_bound_rhs_1: lp_bound_rhs_1
                             , self.lp_bound_rhs_2: lp_bound_rhs_2})            
        else:
            w_input, b_input, w_output, b_output, w_hidden, b_hidden = session.run(
                [self.weight_input, self.bias_input, self.weight_output, self.bias_output, self.weight_hidden, self.bias_hidden]
                , feed_dict={self.state_lhs: data_lhs
                             , self.state_rhs_1: data_rhs_1
                             , self.state_rhs_2: data_rhs_2
                             , self.mask: data_mask
                             })            
        print("input layer:", "%.4f"%np.mean(w_input), "%.4f"%np.mean(b_input))    
        for w,b in zip(w_hidden,b_hidden):
            print("hidden layer:", "%.4f"%np.mean(w), "%.4f"%np.mean(b))    
        print("output layer:", "%.4f"%np.mean(w_output), "%.4f"%np.mean(b_output))    
        #print("weights = ", result_weights, " \nbias = ", result_bias)
    #EOC

#generate a batch of data for training and/or validation
def generate_batch(param, sample_generator = None):
    batch_size = param["batch_size"]
    num_product = param["num_product"]
    num_steps = param["num_steps"]
    product_prob = param["product_prob"]
    num_nights = param["num_nights"]
    capacity = param["capacity"]
    product_resource_map = param["product_resource_map"]
    product_revenue = param["product_revenue"]
    debug_lp = param["debug_lp"]      
    
    if sample_generator == None:
        data_lhs_0 = np.random.choice(capacity+1, [batch_size, num_nights])
        time_lhs = np.random.choice(range(1,num_steps), [batch_size,1])                                       
    else:
        data_lhs_0, time_lhs = sample_generator.next()    
    
    lpb_lhs = np.ones(batch_size)
    lpb_rhs2 = np.ones(batch_size)
    lpb_rhs1 = np.ones([batch_size, num_product])
    
    if debug_lp:
        if np.any(time_lhs<-1e-6):
            with open("lp.error.log","a") as fout:
                print("batch generation error", file=fout)
                print(time_lhs, file=fout)
        lpb_lhs = np.asarray([lp(data_lhs_0[b].astype(np.float32)
                                , (time_lhs[b]*product_prob).astype(np.float32)
                                , param) for b in range(batch_size)])
    
    #generate data for V(s-a(p),t-1)
    #batch x product x night(state)
    resource_consumed = np.tile(product_resource_map, (batch_size,1,1))
    rhs1 = np.repeat(data_lhs_0.flatten(), num_product)
    rhs1 = np.reshape(rhs1, (batch_size, num_nights, -1))
    rhs1 = np.swapaxes(rhs1, 1,2) - resource_consumed
    mask = (1-np.any(rhs1<0, axis=2)).astype(int)        
    rhs1 = np.maximum(rhs1, 0)             
    
    #t-1
    time_rhs = time_lhs-1
    if debug_lp:            
        lpb_rhs2 = np.asarray([lp(data_lhs_0[b].astype(np.float32)
                                , (time_rhs[b]*product_prob).astype(np.float32)
                                , param
                                ) for b in range(batch_size)])
    time_rhs = np.reshape(np.repeat(np.ravel(time_rhs),num_product), (batch_size,num_product,-1))                

    if debug_lp:
        lpb_rhs1 = [ lp(rhs1[b][p]
                        , time_rhs[b][p]*product_prob
                        , param
                        ) for b,p in itertools.product(range(batch_size), range(num_product)) ]                
        lpb_rhs1 = np.reshape(lpb_rhs1, [batch_size,-1])
    
    
    #scaling and stacking
    lhs = np.hstack([np.divide(data_lhs_0, capacity), np.divide(time_lhs, num_steps)])
    rhs1 = np.concatenate([np.divide(rhs1, capacity), np.divide(time_rhs, num_steps)], axis=2)  
    rhs2 = np.hstack([np.divide(data_lhs_0, capacity), np.divide(time_lhs-1, num_steps)])
    
    return lhs, rhs1, rhs2, mask, lpb_lhs, lpb_rhs1, lpb_rhs2
    #EOF 
    
#generate validation data batch with fix time
def generate_batch_fix_time(param):
    batch_size = param["batch_size"]
    num_product = param["num_product"]
    num_steps = param["num_steps"]
    product_prob = param["product_prob"]
    num_nights = param["num_nights"]
    capacity = param["capacity"]
    product_resource_map = param["product_resource_map"]
    product_revenue = param["product_revenue"]
    debug_lp = param["debug_lp"]      
    #generate monotonic state sequence
    booking = np.random.choice(range(0,2), [batch_size, num_nights])
    for k in range(1, batch_size):
        booking[k] = booking[k] + booking[k-1]
    #data_lhs_0 = np.random.choice(capacity+1, [batch_size, num_nights])
    data_lhs_0 = np.random.choice(capacity+1, [num_nights]) + booking
    data_lhs_0 = np.minimum(data_lhs_0, capacity)
    #fix time
    time_lhs = np.random.choice(range(1,num_steps))*np.ones([batch_size,1])
    
    lpb_lhs = np.ones(batch_size)
    lpb_rhs2 = np.ones(batch_size)
    lpb_rhs1 = np.ones([batch_size, num_product])
    
    if debug_lp:
        lpb_lhs = np.asarray([lp(data_lhs_0[b].astype(np.float32)
                            , (time_lhs[b]*product_prob).astype(np.float32)
                            , param
                            ) for b in range(batch_size)])
    
    #generate data for V(s-a(p),t-1)
    #batch x product x night(state)
    resource_consumed = np.tile(product_resource_map, (batch_size,1,1))
    rhs1 = np.repeat(data_lhs_0.flatten(), num_product)
    rhs1 = np.reshape(rhs1, (batch_size, num_nights, -1))
    rhs1 = np.swapaxes(rhs1, 1,2,) - resource_consumed
    mask = (1-np.any(rhs1<0, axis=2)).astype(int)        
    rhs1 = np.maximum(rhs1, 0)             
    
    #t-1
    time_rhs = time_lhs-1
    if debug_lp:            
        lpb_rhs2 = np.asarray([lp(data_lhs_0[b].astype(np.float32)
                                , (time_rhs[b]*product_prob).astype(np.float32)
                                , param
                                ) for b in range(batch_size)])
    time_rhs = np.reshape(np.repeat(np.ravel(time_rhs),num_product), (batch_size,num_product,-1))                

    if debug_lp:
        lpb_rhs1 = [ lp(rhs1[b][p]
                        , time_rhs[b][p]*product_prob
                        , param
                        ) for b,p in itertools.product(range(batch_size), range(num_product)) ]                
        lpb_rhs1 = np.reshape(lpb_rhs1, [batch_size,-1])
    
    
    #scaling and stacking
    lhs = np.hstack([np.divide(data_lhs_0, capacity), np.divide(time_lhs, num_steps)])
    rhs1 = np.concatenate([np.divide(rhs1, capacity), np.divide(time_rhs, num_steps)], axis=2)  
    rhs2 = np.hstack([np.divide(data_lhs_0, capacity), np.divide(time_lhs-1, num_steps)])
    
    return lhs, rhs1, rhs2, mask, lpb_lhs, lpb_rhs1, lpb_rhs2
    #EOF     
    
#generate validation batch data for monotonicity checking 
# with constant state
def generate_batch_t0(param):  
    batch_size = param["batch_size"]
    num_product = param["num_product"]
    num_steps = param["num_steps"]
    product_prob = param["product_prob"]
    num_nights = param["num_nights"]
    capacity = param["capacity"]
    product_resource_map = param["product_resource_map"]
    product_revenue = param["product_revenue"]
    debug_lp = param["debug_lp"]      
    #generate one state
    data_lhs_0 = np.random.choice(capacity+1, [num_nights])
    #and fix it
    data_lhs_0 = np.tile(data_lhs_0, (batch_size,1))
    # sort time
    time_lhs = np.reshape(np.sort(np.random.choice(range(1,num_steps), [batch_size,1]), axis=None), [batch_size,-1])
    
    lpb_lhs = np.ones(batch_size)
    lpb_rhs2 = np.ones(batch_size)
    lpb_rhs1 = np.ones([batch_size, num_product])    
    
    if debug_lp:
        lpb_lhs = np.asarray([lp(data_lhs_0[b].astype(np.float32)
                                , (time_lhs[b]*product_prob).astype(np.float32)
                                , param
                                ) for b in range(batch_size)])    
    
    #generate data for V(s-a(p),t-1)
    #batch x product x night(state)
    resource_consumed = np.tile(product_resource_map, (batch_size,1,1))
    rhs1 = np.repeat(data_lhs_0.flatten(), num_product)
    rhs1 = np.reshape(rhs1, (batch_size, num_nights, -1))
    rhs1 = np.swapaxes(rhs1, 1,2,) - resource_consumed
    mask = (1-np.any(rhs1<0, axis=2)).astype(int)        
    rhs1 = np.maximum(rhs1, 0)             
    
    #t-1
    time_rhs = time_lhs-1
    if debug_lp:            
        lpb_rhs2 = np.asarray([lp(data_lhs_0[b].astype(np.float32)
                                , (time_rhs[b]*product_prob).astype(np.float32)
                                , param
                                ) for b in range(batch_size)])
    time_rhs = np.reshape(np.repeat(np.ravel(time_rhs),num_product), (batch_size,num_product,-1))                

    if debug_lp:
        lpb_rhs1 = [ lp(rhs1[b][p], time_rhs[b][p]*product_prob, param) 
                        for b,p in itertools.product(range(batch_size), range(num_product)) ]                
        lpb_rhs1 = np.reshape(lpb_rhs1, [batch_size,-1])
    
    #scaling and stacking
    lhs = np.hstack([np.divide(data_lhs_0, capacity), np.divide(time_lhs, num_steps)])
    rhs1 = np.concatenate([np.divide(rhs1, capacity), np.divide(time_rhs, num_steps)], axis=2)  
    rhs2 = np.hstack([np.divide(data_lhs_0, capacity), np.divide(time_lhs-1, num_steps)])
    
    return lhs, rhs1, rhs2, mask, lpb_lhs, lpb_rhs1, lpb_rhs2
    #EOF     
      
#build a class for stratified sampling
class sample_generation:    
    def __init__(self, param):    
        self.param = param
        self.build()
        
    def build(self):
        #param = param
        batch_size = self.param["batch_size"]
        #num_product = param["num_product"]
        num_steps = self.param["num_steps"]
        product_prob = self.param["product_prob"]
        num_nights = self.param["num_nights"]
        capacity = self.param["capacity"]
        product_resource_map = self.param["product_resource_map"]
        product_revenue = self.param["product_revenue"]
        num_product = self.param["num_product"]
            
        #generate demand for all tsteps
        demand = np.random.choice(range(num_product)
                                        , size=(num_steps, batch_size)
                                        , p=product_prob
                                     )    
        self.result = np.zeros([num_steps, batch_size, num_nights])
        # initial state
        state = np.ones([batch_size, num_nights])*capacity        
        for tstep in range(num_steps):              
            resource = np.stack([product_resource_map[p] for p in demand[tstep]])
            #admit = np.ones((batch_size, 1))
            admit = 1-np.any((state-resource)<-1e-6, axis=1).astype(int)
            state = state - np.multiply(resource, np.reshape(admit, [batch_size,1]))
            self.result[tstep] = state        
        self.num_steps = num_steps
        self.order = np.arange(num_steps)
        self.tstep = 0
        np.random.shuffle(self.order)
        
    def next(self):
        batch_size = self.param["batch_size"]
        if self.tstep >= self.num_steps:
            # what if we run out of sample? just shuffle again
            # and start over            
            self.build()
        tstep = self.order[self.tstep]
        self.tstep += 1        
        if tstep < 0:
            with open("lp.error.log","a") as fout:
                print("sample generator error", file=fout)
                print(tstep, file=fout)
        return self.result[tstep], np.full([batch_size,1], tstep)    
        
