#the purpose of this file is to generate a simple functional
# approximator to V(s,t)

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import time

#we also need to define boundary conditions for V(s,0) = 0
def generate_boundary_error(s, weights, bias):
    weights_t0 = tf.multiply(weights, tf.constant(np.concatenate([np.ones(num_nights), np.zeros(1)]), dtype=tf.float32))
    v_t0 = tf.reduce_sum(tf.multiply(state_lhs, weights_t0), axis=1) + bias
    
    lambda_t0 = 1.0
    boundary_t0 = tf.multiply(tf.reduce_mean(tf.multiply(v_t0,v_t0)), tf.constant(lambda_t0, dtype=tf.float32))
    
    #V(0,t) = 0
    weights_s0 = tf.multiply(weights, tf.constant(np.concatenate([np.zeros(num_nights), np.ones(1)]), dtype=tf.float32))
    v_s0 = tf.reduce_sum(tf.multiply(state_lhs, weights_s0), axis=1) + bias
    
    lambda_s0 = 1.0
    boundary_s0 = tf.multiply(tf.reduce_mean(tf.multiply(v_s0,v_s0)), tf.constant(lambda_s0, dtype=tf.float32))
    return boundary_t0 + boundary_s0

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

product_revenue = 10*np.random.uniform(size=[num_product])
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
state_lhs = tf.placeholder(tf.float32, [batch_size, dim_state_space])
#V(s-a(p),t-1)
state_rhs_1 = tf.placeholder(tf.float32, [batch_size, num_product, dim_state_space])
#V(s,t-1)
state_rhs_2 = tf.placeholder(tf.float32, [batch_size, dim_state_space])
mask = tf.placeholder(tf.float32, [batch_size, num_product])
#probability

#define linear approximation parameters
weights = tf.Variable(np.random.uniform(size=[dim_state_space]), dtype=tf.float32)
bias = tf.Variable(np.random.uniform(size=1), dtype=tf.float32)

#define LHS
value_lhs = tf.reduce_sum(tf.multiply(state_lhs, weights), axis=1) + bias

#V(s,t-1) as a matrix of batch x 1
value_rhs_2 = tf.reduce_sum(tf.multiply(state_rhs_2, weights), axis=1) + bias

# this is a long definition for the sum max calculation done in multiple steps
#V(s-a(p),t-1) for every p, dimension is batch x product
value_rhs_1 = tf.reduce_sum(tf.multiply(state_rhs_1, weights), axis=2) + bias
#V(s-a(p),t-1) - V(s,t-1) + r(p)
value_rhs_1 = value_rhs_1 - tf.reshape(value_rhs_2, [batch_size,-1]) + tf.constant(product_revenue, dtype=tf.float32)
# max(x,0)
value_rhs_1 = tf.maximum(value_rhs_1, tf.constant(0.0))
#we need the mask here because certain products are unsellable given
#a certain state. To implement this logic, we do two things:
# 1. setting mask = 0 for such state/product combination
# 2. in data preparation setting that state to 0 
#in this way, no error should come up in approximator
#and no impact on gradient estimator
value_rhs_1 = tf.multiply(value_rhs_1, mask)
#prob*max
value_rhs_1 = tf.multiply(value_rhs_1
                        , tf.constant(product_prob
                                      , dtype=tf.float32))
#sum (prob*max)
value_rhs_1 = tf.reduce_sum(value_rhs_1, axis=1)
#V(s,t-1) + sum pr*max(*)
value_rhs = value_rhs_1 + value_rhs_2

bellman_error = value_lhs-value_rhs
bellman_error = tf.multiply(bellman_error,bellman_error)

boundary_error = generate_boundary_error(state_lhs, weights, bias)

#training loss
loss = tf.reduce_mean(bellman_error) + boundary_error

train_step = tf.train.AdagradOptimizer(0.3).minimize(loss)    

num_batches = 1000
with tf.Session() as sess:    
    sess.run(tf.global_variables_initializer())
    for batch in range(num_batches):
        #generate data for LHS
        data_lhs_0 = np.random.choice(capacity+1, [batch_size, num_nights])
        time_lhs = np.random.choice(range(1,num_steps), [batch_size,1])        
        data_rhs_2 = np.hstack([data_lhs_0, time_lhs-1])
        data_lhs = np.hstack([data_lhs_0, time_lhs])
        
        #derive RHS
        # after consuming every product, what's the RHS
        #batch x product x night(state)
        #resource_consumed = np.swapaxes(np.tile(product_resource_map, (batch_size,1,1)), 1,2)
        resource_consumed = np.tile(product_resource_map, (batch_size,1,1))
        #data_rhs_1 = np.reshape(data_lhs_0, (batch_size,num_product,-1)) - resource_consumed
        data_rhs_1 = np.repeat(data_lhs_0.flatten(), num_product)
        data_rhs_1 = np.reshape(data_rhs_1, (batch_size, num_nights, -1))
        data_rhs_1 = np.swapaxes(data_rhs_1, 1,2,) - resource_consumed
        data_mask = (1-np.any(data_rhs_1<0, axis=2)).astype(int)        
        data_rhs_1 = np.maximum(data_rhs_1, 0)
        time_rhs = time_lhs-1
        time_rhs = np.reshape(np.repeat(np.ravel(time_rhs),num_product), (batch_size,num_product,-1))
        data_rhs_1 = np.concatenate([data_rhs_1, time_rhs], axis=2)  
        
        #we will have to run the session twice since there is no way
        # to ensure it is executed in a pre-determined order per
        # https://github.com/tensorflow/tensorflow/issues/13133
        # this is the training step
        sess.run(train_step
                , feed_dict={state_lhs: data_lhs,
                             state_rhs_1: data_rhs_1,
                             state_rhs_2: data_rhs_2,
                             mask: data_mask
                             })
        # this is simply forward calculation        
        if 1 and batch % 100 == 0:    
            result_loss, result_weights, result_bias, result_lhs, result_rhs_1, result_rhs_2 = sess.run(
                [loss, weights, bias, value_lhs, value_rhs_1, value_rhs_2]
                , feed_dict={state_lhs: data_lhs,
                             state_rhs_1: data_rhs_1,
                             state_rhs_2: data_rhs_2,
                             mask: data_mask
                             })    
            print("batch = ", batch, " loss = %.2f"%result_loss)
            print("weights = ", result_weights, " bias = ", result_bias)

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