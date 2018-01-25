#the purpose of this file is to generate a simple functional
# approximator to V(s,t)

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import time



class error_model_simple_nn:
    def __init__(self):
        #inputs
        self.state_lhs = tf.placeholder(tf.float32, [batch_size, dim_state_space])
        #V(s-a(p),t-1)
        self.state_rhs_1 = tf.placeholder(tf.float32, [batch_size, num_product, dim_state_space])
        #V(s,t-1)
        self.state_rhs_2 = tf.placeholder(tf.float32, [batch_size, dim_state_space])
        self.mask = tf.placeholder(tf.float32, [batch_size, num_product])                
        
        self.hidden = 64
        self.output = 1
        self.value_lhs = tf.constant(0.0, dtype=tf.float32)
        self.value_rhs_1 = tf.constant(0.0, dtype=tf.float32)
        self.value_rhs_2 = tf.constant(0.0, dtype=tf.float32)
        self.weight_input = tf.Variable(np.random.uniform(size=[dim_state_space, self.hidden]), dtype=tf.float32)
        self.bias_input = tf.Variable(np.random.uniform(size=[self.hidden]), dtype=tf.float32)
        self.weight_output = tf.Variable(np.random.uniform(size=[self.hidden,1]), dtype=tf.float32)
        self.bias_output = tf.Variable(np.random.uniform(size=[1,1]), dtype=tf.float32)
        self.lambda_s0 = 1e-3
        self.lambda_t0 = 1e-3
        
        num_hidden_layer = 5
        hidden_units = [64] * num_hidden_layer
        
        #with hidden units we can initialize size
        self.weight_hidden = [tf.Variable(np.random.uniform(size=[din,dout]), dtype=tf.float32) for din,dout in zip(hidden_units, hidden_units[1:])]
        self.bias_hidden = [tf.Variable(np.random.uniform(size=[dout]), dtype=tf.float32) for din,dout in zip(hidden_units, hidden_units[1:])]        

        self.loss = self.generate_bellman_error_deep() \
                        + self.generate_boundary_error_deep()
        self.train_step =tf.train.AdagradOptimizer(0.3).minimize(self.loss)
        
    def build_network(self, state_input, weights, biases):
        state = state_input
        for w,b in zip(weights, biases):
            state = tf.nn.sigmoid(tf.matmul(state, w) + b)
        return state

    def generate_bellman_error_deep(self):
        #temporary inputs, removing later
#        state_lhs = tf.placeholder(tf.float32, [batch_size, dim_state_space])
#        state_rhs_1 = tf.placeholder(tf.float32, [batch_size, num_product, dim_state_space])
#        state_rhs_2 = tf.placeholder(tf.float32, [batch_size, dim_state_space])
#        mask = tf.placeholder(tf.float32, [batch_size, num_product])
#        weight_input = tf.Variable(np.random.uniform(size=[dim_state_space, hidden]), dtype=tf.float32)
#        bias_input = tf.Variable(np.random.uniform(size=[hidden]), dtype=tf.float32)
#        weight_output = tf.Variable(np.random.uniform(size=[hidden,1]), dtype=tf.float32)
#        bias_output = tf.Variable(np.random.uniform(size=[1,1]), dtype=tf.float32)                
        
        weights = [self.weight_input] + self.weight_hidden + [self.weight_output]
        biases = [self.bias_input] + self.bias_hidden + [self.bias_output]
        #V(s,t)
        V_s_t = self.build_network(self.state_lhs, weights, biases)
        #V(s,t-1)
        V_s_tm1 = self.build_network(self.state_rhs_2, weights, biases)
        #V(s-a(p),t-1)
        V_s1_tm1 = self.build_network(tf.reshape(self.state_rhs_1,[-1, dim_state_space]), weights, biases)
        V_s1_tm1 = tf.reshape(V_s1_tm1, [batch_size,-1])             
        
        value_lhs = V_s_t
        value_rhs_2 = V_s_tm1
        
        #V(s-a(p),t-1) - V(s,t-1) + r(p)
        value_rhs_1 = V_s1_tm1 - tf.reshape(V_s_tm1, [batch_size,-1]) + tf.constant(product_revenue, dtype=tf.float32)
        # max(,0)
        value_rhs_1 = tf.maximum(value_rhs_1, tf.constant(0.0))
        # mask unsellable product
        value_rhs_1 = tf.multiply(value_rhs_1, self.mask)
        # multiply by probability
        value_rhs_1 = tf.multiply(value_rhs_1
                                , tf.constant(product_prob
                                              , dtype=tf.float32))        
        # sum
        value_rhs_1 = tf.reshape(tf.reduce_sum(value_rhs_1, axis=1), [batch_size,-1])        
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
        
    def generate_bellman_error_shallow(self):
        #V(s,t)
        hidden_lhs = tf.nn.sigmoid(tf.matmul(self.state_lhs, self.weight_input) + self.bias_input)
        #output self.build_network: it's unclear what output self.build_network makes sense for
        #a continuous output at this point. we assume it is linear
        value_lhs = tf.matmul(hidden_lhs, self.weight_output) + self.bias_output
        #print(value_lhs)
        
        #V(s,t-1)
        hidden_rhs_2 = tf.nn.sigmoid(tf.matmul(self.state_rhs_2, self.weight_input) + self.bias_input)
        value_rhs_2 = tf.matmul(hidden_rhs_2, self.weight_output) + self.bias_output
        #print(value_rhs_2 )
        
        #V(s-a(p),t-1)
        hidden_rhs_1 = tf.nn.sigmoid(tf.tensordot(self.state_rhs_1, self.weight_input, axes=[[2],[0]]) + self.bias_input)        
        value_rhs_1 = tf.reshape(tf.tensordot(hidden_rhs_1, self.weight_output, axes=[[2],[0]]) + self.bias_output, [batch_size,-1])        
        
        value_rhs_1 = value_rhs_1 - tf.reshape(value_rhs_2, [batch_size,-1]) + tf.constant(product_revenue, dtype=tf.float32)
        value_rhs_1 = tf.maximum(value_rhs_1, tf.constant(0.0))
        value_rhs_1 = tf.multiply(value_rhs_1, self.mask)
        value_rhs_1 = tf.multiply(value_rhs_1
                                , tf.constant(product_prob
                                              , dtype=tf.float32))        
        value_rhs_1 = tf.reshape(tf.reduce_sum(value_rhs_1, axis=1), [batch_size,-1])        
        value_rhs = value_rhs_1 + value_rhs_2        
        #print(value_rhs)
        
        bellman_error = value_lhs-value_rhs
        #print(bellman_error)
        self.bellman_error = tf.reduce_mean(tf.multiply(bellman_error,bellman_error))
        self.value_lhs = value_lhs 
        self.value_rhs_1 = value_rhs_1
        self.value_rhs_2 = value_rhs_2
        #print(self.bellman_error)
        return self.bellman_error
    
    def generate_boundary_error_deep(self): 
        #V(s,0) = 0
        
        mask_t0 = tf.reshape(tf.constant(np.concatenate([np.ones(num_nights), np.zeros(1)]), dtype=tf.float32), [dim_state_space, -1])
        weights_t0 = tf.multiply(self.weight_input, mask_t0)                 
        weights = [weights_t0] + self.weight_hidden + [self.weight_output]
        biases = [self.bias_input] + self.bias_hidden + [self.bias_output]                
        #V(s,t=0)
        v_t0 = self.build_network(self.state_lhs, weights, biases)
        boundary_t0 = tf.multiply(tf.reduce_mean(tf.multiply(v_t0,v_t0)), tf.constant(self.lambda_t0, dtype=tf.float32))
        
        mask_s0 = tf.reshape(tf.constant(np.concatenate([np.zeros(num_nights), np.ones(1)]), dtype=tf.float32), [dim_state_space, -1])
        weights_s0 = tf.multiply(self.weight_input, mask_s0)
        weights = [weights_s0] + self.weight_hidden + [self.weight_output]        
        #V(s=0,t)
        v_s0 = self.build_network(self.state_lhs, weights, biases)                
        boundary_s0 = tf.multiply(tf.reduce_mean(tf.multiply(v_s0,v_s0)), tf.constant(self.lambda_s0, dtype=tf.float32))
        self.boundary_error = boundary_t0 + boundary_s0
        return self.boundary_error   
    
    def generate_boundary_error_shallow(self): 
        #V(s,0) = 0
        mask_t0 = tf.reshape(tf.constant(np.concatenate([np.ones(num_nights), np.zeros(1)]), dtype=tf.float32), [dim_state_space, -1])
        weights_t0 = tf.multiply(self.weight_input, mask_t0) 
        hidden_lhs = tf.nn.sigmoid(tf.matmul(self.state_lhs, weights_t0) + self.bias_input)
        #output self.build_network: it's unclear what output self.build_network makes sense for
        #a continuous output at this point. we assume it is linear
        v_t0 = tf.matmul(hidden_lhs, self.weight_output) + self.bias_output                        
        print(v_t0)
        boundary_t0 = tf.multiply(tf.reduce_mean(tf.multiply(v_t0,v_t0)), tf.constant(self.lambda_t0, dtype=tf.float32))
        
        mask_s0 = tf.reshape(tf.constant(np.concatenate([np.zeros(num_nights), np.ones(1)]), dtype=tf.float32), [dim_state_space, -1])
        weights_s0 = tf.multiply(self.weight_input, mask_s0)
        hidden_lhs = tf.nn.sigmoid(tf.matmul(self.state_lhs, weights_s0) + self.bias_input)
        v_s0 = tf.matmul(hidden_lhs, self.weight_output) + self.bias_output        
                
        boundary_s0 = tf.multiply(tf.reduce_mean(tf.multiply(v_s0,v_s0)), tf.constant(self.lambda_s0, dtype=tf.float32))
        self.boundary_error = boundary_t0 + boundary_s0
        return self.boundary_error   

    def train(self, session, data_lhs, data_rhs_1, data_rhs_2, data_mask):
        session.run(self.train_step
                , feed_dict={self.state_lhs: data_lhs,
                             self.state_rhs_1: data_rhs_1,
                             self.state_rhs_2: data_rhs_2,
                             self.mask: data_mask
                             })
        
    def read_loss(self, session, data_lhs ,data_rhs_1 ,data_rhs_2 ,data_mask):
        result_loss, result_bellman, result_boundary = session.run(
                [self.loss, self.bellman_error, self.boundary_error]
                , feed_dict={self.state_lhs: data_lhs,
                             self.state_rhs_1: data_rhs_1,
                             self.state_rhs_2: data_rhs_2,
                             self.mask: data_mask
                             })            
        print("loss = %.6f"%result_loss, "bellman error = %.6f"%result_bellman, "boundary error = %.6f"%result_boundary)
        
    def read_param(self, session, data_lhs,data_rhs_1,data_rhs_2,data_mask):
        #read we have set up the network and it is running
        #all we need to do is to refer to objects we created
        # and are interested
        result_lhs, result_rhs_1, result_rhs_2, result_weights, result_bias = session.run(
                [self.value_lhs, self.value_rhs_1, self.value_rhs_2, self.weight_input, self.bias_input]
                , feed_dict={self.state_lhs: data_lhs,
                             self.state_rhs_1: data_rhs_1,
                             self.state_rhs_2: data_rhs_2,
                             self.mask: data_mask
                             })            
        print("weights = ", result_weights, " \nbias = ", result_bias)
    #EOC
    
#general initialization
ts = time.time()
ops.reset_default_graph()
np.set_printoptions(precision=4)
np.random.seed(4321)

#business parameter initialization
num_nights = 14
capacity = 100
# product zero is the no-revenue no resource product
# added for simplicity
product_null = 0
# unfortunately, to avoid confusion we need to add a fairly 
# complex product matrix
# if there are N nights, there are N one-night product from 
# 1 to N; there are also N-1 two-night products from N+1 to 2N-1
num_product = num_nights*2
product_resource_map = np.zeros((num_product, num_nights))
for i in range(1,num_nights):
    product_resource_map[i][i-1] = 1.0
    product_resource_map[i][i] = 1.0
for i in range(0,num_nights):    
    product_resource_map[i+num_nights][i] = 1.0
#product_resource_map[num_product-1][num_nights-1] = 1.0

product_revenue = 100*np.random.uniform(size=[num_product])
product_revenue[product_null] = 0
#total demand
product_demand = np.random.uniform(size=[num_product])*capacity
product_demand[product_null]  = 0

num_steps = int(np.sum(product_demand)/0.01)

#arrival rate (including)
product_prob = np.divide(product_demand,num_steps)
product_prob[0] = 1.0 - np.sum(product_prob)

#computational graph generation

#define a state (in batch) and a linear value function
batch_size = 64
#LHS is the value function for current state at time t
#for each state, we need num_nights real value inputs for available
# inventory, and +1 for time
dim_state_space = num_nights+1

#tensorflow model inputs (or really state space samples)
#V(s,t)
#try neural network model: input->hidden->output



#define linear approximation model
#model = error_model_linear()
model = error_model_simple_nn()

num_batches = 5000
with tf.Session() as sess:    
    sess.run(tf.global_variables_initializer())
    for batch in range(num_batches):
        #generate data for LHS
        data_lhs_0 = np.random.choice(capacity+1, [batch_size, num_nights])
        time_lhs = np.random.choice(range(1,num_steps), [batch_size,1])                               
        #generate data for RHS
        #batch x product x night(state)
        resource_consumed = np.tile(product_resource_map, (batch_size,1,1))
        data_rhs_1 = np.repeat(data_lhs_0.flatten(), num_product)
        data_rhs_1 = np.reshape(data_rhs_1, (batch_size, num_nights, -1))
        data_rhs_1 = np.swapaxes(data_rhs_1, 1,2,) - resource_consumed
        data_mask = (1-np.any(data_rhs_1<0, axis=2)).astype(int)        
        data_rhs_1 = np.maximum(data_rhs_1, 0)
        #t-1
        time_rhs = time_lhs-1
        time_rhs = np.reshape(np.repeat(np.ravel(time_rhs),num_product), (batch_size,num_product,-1))        
        
        #scaling and stacking
        data_lhs = np.hstack([np.divide(data_lhs_0, capacity), np.divide(time_lhs, num_steps)])
        data_rhs_1 = np.concatenate([np.divide(data_rhs_1, capacity), np.divide(time_rhs, num_steps)], axis=2)  
        data_rhs_2 = np.hstack([np.divide(data_lhs_0, capacity), np.divide(time_lhs-1, num_steps)])
                
        #we will have to run the session twice since tensorflow does
        # not ensure all tasks are executed in a pre-determined order per
        # https://github.com/tensorflow/tensorflow/issues/13133
        # this is the training step
        model.train(sess,data_lhs,data_rhs_1,data_rhs_2,data_mask)
        # statistics accumulation
        if 1 and batch % 100 == 0:              
            print("batch = ", batch)
            model.read_loss(sess, data_lhs, data_rhs_1, data_rhs_2, data_mask)

print("total program time = %.2f seconds" % (time.time()-ts))

#in debug model, build np version of lhs and rhs
#np_lhs = np.sum(np.multiply(data_lhs, result_weights),axis=1) + result_bias
#np_rhs_2 = np.sum(np.multiply(data_rhs_2, result_weights),axis=1) + result_bias
#
#np_rhs_1 = np.sum(np.multiply(data_rhs_1, result_weights), axis=2) + result_bias
#np_rhs_1 = np_rhs_1 - np.reshape(np_rhs_2, [batch_size,-1]) + np.asarray(product_revenue)
#np_rhs_1 = np.maximum(np_rhs_1, 0.0)
#np_rhs_1 = np.multiply(np_rhs_1, data_mask)
#np_rhs_1 = np.multiply(np_rhs_1, product_prob)
#np_rhs_1 = np.sum(np_rhs_1, axis=1)