#the purpose of this file is to generate a simple functional 
# approximator to V(s,t)

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import time

ts = time.time()

ops.reset_default_graph()
np.set_printoptions(precision=4)
np.random.seed(4321)

num_nights = 4
capacity = 10
# the true number of products is num_nights-1 because every products 
# take a 2-night stay
# product zero is the no-revenue no resource product
# added for simplicity
product_null = 0

num_product = num_nights+1
product_resource_map = np.zeros((num_product, num_nights))
for i in range(1,num_product-1):
    product_resource_map[i][i-1] = 1.0
    product_resource_map[i][i] = 1.0
product_resource_map[num_product-1][num_nights-1] = 1.0

product_revenue = np.random.uniform(size=[num_product])
product_revenue[product_null] = 0
#total demand
product_demand = np.random.uniform(size=[num_product])*capacity
product_demand[product_null]  = 0

num_steps = int(np.sum(product_demand)/0.01)

#arrival rate (including)
product_prob = np.divide(product_demand,num_steps)
product_prob[0] = 1.0 - np.sum(product_prob)

#define a state (in batch) and a linear value function
batch_size = 5

#LHS is the value function for current state at time t
#for each state, we need num_nights real value inputs for available
# inventory, and +1 for time
dim_state_space = num_nights+1

state_lhs = tf.placeholder(tf.float32, [batch_size, dim_state_space])
#RHS is the value function for relevant states at time t-1
state_rhs_1 = tf.placeholder(tf.float32, [batch_size, num_product, dim_state_space])
mask = tf.placeholder(tf.float32, [batch_size, num_product])
#probability

#define linear approximation parameters
weights = tf.Variable(np.random.uniform(size=[dim_state_space]), dtype=tf.float32)
bias = tf.Variable(np.random.uniform(size=1), dtype=tf.float32)

#define LHS
value_lhs = tf.multiply(state_lhs, weights)
value_lhs = tf.reduce_sum(value_lhs, axis=1) + bias

#define RHS
value_rhs = tf.multiply(state_rhs_1, weights)
#we need the mask here because certain products are unsellable given
#a certain state. To implement this logic, we do two things:
# 1. setting mask = 0 for such state/product combination
# 2. in data preparation setting that state to 0 
#in this way, no error should come up in approximator
#and no impact on gradient estimator
#value_rhs = tf.multiply(mask, tf.reduce_sum(value_rhs, axis=2) + bias)
value_rhs = tf.reduce_sum(value_rhs, axis=2) + bias
value_rhs = tf.maximum(value_rhs + 
                       tf.constant(product_revenue, dtype=tf.float32) - 
                       tf.reshape(value_lhs, [-1,1]), tf.constant(0.0))
value_rhs = tf.multiply(value_rhs, mask)
value_rhs = tf.multiply(value_rhs
                        , tf.constant(product_prob
                                      , dtype=tf.float32))
value_rhs = tf.reduce_sum(value_rhs, axis=1)

bellman_error = value_lhs-value_rhs
loss = tf.reduce_mean(tf.multiply(bellman_error,bellman_error))

train_step = tf.train.AdagradOptimizer(0.3).minimize(loss)    

#gw = tf.gradients(loss, weights)
#gb = tf.gradients(loss, bias)

num_batches = 2500
with tf.Session() as sess:    
    sess.run(tf.global_variables_initializer())
    for batch in range(num_batches):
        #generate data for LHS
        #state_lhs = tf.placeholder(tf.float32, [batch_size, dim_state_space])
        data_lhs = np.random.choice(capacity+1, [batch_size, num_nights])
        time_lhs = np.random.choice(range(1,num_steps), [batch_size,1])
        data_lhs = np.hstack([data_lhs, time_lhs])
        
        #derive RHS
        # after consuming every product, what's the RHS
        #batch x product x night(state)
        #resource_consumed = np.swapaxes(np.tile(product_resource_map, (batch_size,1,1)), 1,2)
        resource_consumed = np.tile(product_resource_map, (batch_size,1,1))
        data_rhs = np.reshape(data_lhs, (batch_size,num_product,-1)) - resource_consumed
        data_mask = (1-np.any(data_rhs<0, axis=2)).astype(int)
        
        data_rhs = np.maximum(data_rhs ,0)
        time_rhs = time_lhs-1
        time_rhs = np.reshape(np.repeat(np.ravel(time_rhs),num_product), (batch_size,num_product,-1))
        data_rhs = np.concatenate([data_rhs, time_rhs], axis=2)        
        result, result_weights, result_bias, _ = sess.run([loss, weights, bias, train_step]
                , feed_dict={state_lhs: data_lhs,
                             state_rhs_1: data_rhs,
                             mask: data_mask
                             })
        if batch % 100 == 0:    
            print("batch = ", batch, " result = %.2f"%result)
            print("weights = ", result_weights, " bias = ", result_bias)

print("total program time = %.2f seconds" % (time.time()-ts))