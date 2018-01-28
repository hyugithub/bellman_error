#the purpose of this file is to generate a simple functional
# approximator to V(s,t)

import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import time
from ortools.linear_solver import pywraplp
import itertools

def simulation():    
    # we should use the same seed for both fifo and new policy 
    # to reduce variance
    seed0 = seed_simulation    
    np.random.seed(seed0)
    # initial state
    #state_initial = np.ones([batch_size, num_nights])*capacity
    
    for _ in range(1):
        revenue = {}
        revenue["fifo"] = np.zeros(batch_size)
        revenue["dnn"]  = np.zeros(batch_size)
        
        # for each time step, generate demand
        demand = np.random.choice(range(num_product)
                                    , size=(num_steps, batch_size)
                                    , p=product_prob
                                 )
        state = {}
        state["fifo"] = np.ones([batch_size, num_nights])*capacity
        state["dnn"] = np.ones([batch_size, num_nights])*capacity
            
        for s in np.arange(start=num_steps-1, stop=0, step=-1):
            resource = np.stack([product_resource_map[p] for p in demand[s]])
            for pol in policy_list:
                admit = policy[pol].do(state[pol], resource, demand[s], s)
                revenue0 = np.array([product_revenue[p]*admit[b] for b,p in zip(range(batch_size), demand[s])])
                revenue[pol] = revenue[pol] + revenue0
                state[pol] = state[pol] - np.multiply(resource, np.reshape(admit, [batch_size,1]))
        for r1,r2 in zip(revenue["fifo"], revenue["dnn"]):
            print("lift = %.2f"%(r2/r1-1.0))    

def lp(cap_supply, cap_demand, param, return_dual = False):
    #print(cap_supply.shape, cap_demand.shape)
    ts = time.time()
    #loading parameters
    num_product = param["num_product"]    
    num_nights = param["num_nights"]    
    product_resource_map = param["product_resource_map"]    
    product_revenue = param["product_revenue"]    
    product_null = param["product_null"]    

    solver = pywraplp.Solver('LinearExample',
                           pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
    #variables are number of products sold    
    x = [solver.NumVar(0.0, 1.0*cap_demand[p], "".join(["x",str(p)]))
    #x = [solver.NumVar(0.0, 10, "".join(["x",str(p)])) 
            for p in range(num_product)]
    
    #constraints are capacity for each night
    constraints = []   
    for night in range(num_nights):
        #print(cap_supply[night])
        con = solver.Constraint(0.0, float(cap_supply[night]))
        #con = solver.Constraint(0, capacity)
        for p in range(num_product):        
            con.SetCoefficient(x[p], product_resource_map[p][night])
        constraints.append(con)
    
    #objective        
    objective = solver.Objective()
    for p in range(num_product):
        objective.SetCoefficient(x[p], product_revenue[p])        
    objective.SetCoefficient(x[product_null], -1.0)        
    objective.SetMaximization()    
    
    solver.Solve()
    
    dual = np.array([c.dual_value() for c in constraints])
    #print("dual value:")
    #print(dual)
    
    if 0:    
        for p in range(num_product):
            print("p=", p, "price = %2.f"%product_revenue[p], "demand = %.2f"%cap_demand[p], ' allocation = %.2f'%(x[p].solution_value()))
            
        print('Solution = %.2f' % objective.Value())
        sol2 = np.sum([product_revenue[p]*x[p].solution_value() for p in range(num_product)])
        print("sol2 = %.2f" % sol2)
        
        print("total time = %.2f"%(time.time()-ts))    
    
    if return_dual:
        return objective.Value(), dual
    return objective.Value()

class error_model_simple_nn:
    def __init__(self):
        #inputs
        self.state_lhs = tf.placeholder(tf.float32, [batch_size, dim_state_space])
        #V(s-a(p),t-1)
        self.state_rhs_1 = tf.placeholder(tf.float32, [batch_size, num_product, dim_state_space])
        #V(s,t-1)
        self.state_rhs_2 = tf.placeholder(tf.float32, [batch_size, dim_state_space])
        self.mask = tf.placeholder(tf.float32, [batch_size, num_product])                
        
        if debug_lp:
            self.lp_bound_lhs = tf.placeholder(tf.float32, [batch_size])
            self.lp_bound_rhs_1 = tf.placeholder(tf.float32, [batch_size, num_product])
            self.lp_bound_rhs_2 = tf.placeholder(tf.float32, [batch_size])       

        # size of network        
        self.hidden = 64
        
        #roughly speaking, we need to initialize 
        self.init_level = 1.0
        self.init_level_output = 1.0
        
        self.value_lhs = tf.constant(0.0, dtype=tf.float32)
        self.value_rhs_1 = tf.constant(0.0, dtype=tf.float32)
        self.value_rhs_2 = tf.constant(0.0, dtype=tf.float32)
        self.weight_input = tf.Variable(self.init_level*np.random.normal(size=[dim_state_space, self.hidden])
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
        self.lambda_s0 = 1.0
        self.lambda_t0 = 1.0
        
        num_hidden_layer = 5

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
        self.loss = self.generate_bellman_error_deep() \
                        + self.generate_boundary_error_deep()
#        self.loss = self.generate_bellman_error_deep() \
#                        + tf.multiply(tf.constant(0.0, dtype=tf.float32), self.generate_boundary_error_deep())
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
        if debug_lp:
            print(V_s_t, self.lp_bound_lhs)
            V_s_t = tf.multiply(V_s_t, tf.reshape(self.lp_bound_lhs, [batch_size, -1]))
        #V(s,t-1) 
        V_s_tm1 = self.build_network(self.state_rhs_2, weights, biases)
        if debug_lp:
            V_s_tm1 = tf.multiply(V_s_tm1, tf.reshape(self.lp_bound_rhs_2, [batch_size, -1]))
        #V(s-a(p),t-1)
        V_s1_tm1 = self.build_network(tf.reshape(self.state_rhs_1,[-1, dim_state_space]), weights, biases)
        V_s1_tm1 = tf.reshape(V_s1_tm1, [batch_size,-1])   
        if debug_lp:            
            V_s1_tm1 = tf.multiply(V_s1_tm1, self.lp_bound_rhs_1)
#            print(V_s_t)
#            print(V_s_tm1)
#            print(V_s1_tm1)
        
        value_lhs = V_s_t
        self.value = value_lhs
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
        if debug_lp:
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
        if debug_lp:
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
        if debug_lp:
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
        if debug_lp:
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
def generate_batch():
    data_lhs_0 = np.random.choice(capacity+1, [batch_size, num_nights])
    time_lhs = np.random.choice(range(1,num_steps), [batch_size,1])                                       
    
    lpb_lhs = np.ones(batch_size)
    lpb_rhs2 = np.ones(batch_size)
    lpb_rhs1 = np.ones([batch_size, num_product])
    
    if debug_lp:
        lpb_lhs = np.asarray([lp(data_lhs_0[b].astype(np.float32)
                                , (time_lhs[b]*product_prob).astype(np.float32)
                                , conf) for b in range(batch_size)])
    
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
                                , conf
                                ) for b in range(batch_size)])
    time_rhs = np.reshape(np.repeat(np.ravel(time_rhs),num_product), (batch_size,num_product,-1))                

    if debug_lp:
        lpb_rhs1 = [ lp(rhs1[b][p]
                        , time_rhs[b][p]*product_prob
                        , conf
                        ) for b,p in itertools.product(range(batch_size), range(num_product)) ]                
        lpb_rhs1 = np.reshape(lpb_rhs1, [batch_size,-1])
    
    
    #scaling and stacking
    lhs = np.hstack([np.divide(data_lhs_0, capacity), np.divide(time_lhs, num_steps)])
    rhs1 = np.concatenate([np.divide(rhs1, capacity), np.divide(time_rhs, num_steps)], axis=2)  
    rhs2 = np.hstack([np.divide(data_lhs_0, capacity), np.divide(time_lhs-1, num_steps)])
    
    return lhs, rhs1, rhs2, mask, lpb_lhs, lpb_rhs1, lpb_rhs2
    #EOF 
    
#generate validation data batch with fix time
def generate_batch_fix_time():
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
                            , conf
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
                                , conf
                                ) for b in range(batch_size)])
    time_rhs = np.reshape(np.repeat(np.ravel(time_rhs),num_product), (batch_size,num_product,-1))                

    if debug_lp:
        lpb_rhs1 = [ lp(rhs1[b][p]
                        , time_rhs[b][p]*product_prob
                        , conf
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
def generate_batch_t0():    
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
                                , conf
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
                                , conf
                                ) for b in range(batch_size)])
    time_rhs = np.reshape(np.repeat(np.ravel(time_rhs),num_product), (batch_size,num_product,-1))                

    if debug_lp:
        lpb_rhs1 = [ lp(rhs1[b][p], time_rhs[b][p]*product_prob, conf) 
                        for b,p in itertools.product(range(batch_size), range(num_product)) ]                
        lpb_rhs1 = np.reshape(lpb_rhs1, [batch_size,-1])
    
    #scaling and stacking
    lhs = np.hstack([np.divide(data_lhs_0, capacity), np.divide(time_lhs, num_steps)])
    rhs1 = np.concatenate([np.divide(rhs1, capacity), np.divide(time_rhs, num_steps)], axis=2)  
    rhs2 = np.hstack([np.divide(data_lhs_0, capacity), np.divide(time_lhs-1, num_steps)])
    
    return lhs, rhs1, rhs2, mask, lpb_lhs, lpb_rhs1, lpb_rhs2
    #EOF     
    
class policy_fifo():      
    def do(self, s, r, p, tstep):
        #print(type(s), type(r), type(p))
        return (1.0-np.any((s-r)<0, axis=1)).astype(int)
    #EOC
    
class policy_dnn():      
    def do(self, s, r, p, tstep):
        # check resource
        #print(s.shape, p.shape)
        flag = (1.0-np.any((s-r)<0, axis=1)).astype(int)

        #batch preparation -- avoid LP if possible
        lpb_lhs = np.ones(batch_size)
        lpb_rhs = np.ones(batch_size)
        for b, avail, prod in zip(range(batch_size),flag,p):
            #the logic for dnn policy is complicated because of
            #performanc concerns:
            # 1. for any state and product pair, we need to evaluate
            # the difference between V(s,t) and V(s-a(p),t) (this part is 
            # based on p.89 of TPRM book)
            # this means generating all data for LP and other things
            # on the fly
            # 2. because the cost of step 1 is heavy, we should always
            # avoid calling LP and/or DNN if necessary
            # 3. we don't need to call LP/DNN if there is no physical 
            # capacity or for product null
            
            # 4. moreover, we can keep all previously calculated V(s,t)
            # in a hash table so that getting it is fast            
            if avail <= 1e-6 or prod == product_null:
                continue
            #otherwise it is not a null product and there is avail
            # so we must calculate Vs
            #please note that, because of the structure of our design
            #we have to generate lp bounds for all products, not only
            #the product currently being asssessed            
            
            lpb_lhs[b] = lp(s[b].astype(np.float32)
                        , (tstep*product_prob).astype(np.float32)
                        , conf
                        )
            
            lpb_rhs[b] = lp((s[b]-r[b]).astype(np.float32)
                        , (tstep*product_prob).astype(np.float32)
                        , conf
                        )
            #and then we call DNN
        time_lhs = np.full((batch_size,1), tstep)                            
        
        
        #TODO: call NN only once if batch size can vary
        
        #this is V(s,t)
        data_lhs = np.hstack([np.divide(s, capacity), np.divide(time_lhs, num_steps)])        
        V_lhs = model.predict(sess, data_lhs, lpb_lhs)
        # V(s-a(p),t)
        data_rhs = np.hstack([np.divide(s-r, capacity), np.divide(time_lhs, num_steps)])        
        V_rhs = model.predict(sess, data_rhs, lpb_rhs)
        
        #print(p.shape)
        bid_price = np.reshape(np.array([product_revenue[pp] for pp in p]), [batch_size,1])
        bid_price_delta = bid_price - (V_lhs - V_rhs)
        avail = (bid_price_delta >= 0.0).astype(int).flatten()
        
        #print(avail.shape, flag.shape)        
        return np.multiply(avail, flag)
        
        return flag
        #EOF
        
    #EOC        
    
#general initialization
ts = time.time()
ops.reset_default_graph()
np.set_printoptions(precision=4)

#all parameters
conf = dict()

seed_training = 4321
np.random.seed(seed_training)
conf["seed_training"] = seed_training

seed_simulation = 12345
conf["seed_simulation"] = seed_simulation

if sys.platform == "win32":
    model_path = "C:/Users/hyu/Desktop/bellman/model/"
elif sys.platform == "linux":
    model_path = "/home/ubuntu/model/"
else:
    model_path = ""
    
fname_output_model = model_path+"dp.ckpt"

debug_lp = 1
conf["debug_lp"] = debug_lp

#business parameter initialization
num_nights = 14
conf["num_nights"] = num_nights
capacity = 100
conf["capacity"] = capacity
# product zero is the no-revenue no resource product
# added for simplicity
product_null = 0
conf["product_null"] = product_null
# unfortunately, to avoid confusion we need to add a fairly 
# complex product matrix
# if there are N nights, there are N one-night product from 
# 1 to N; there are also N-1 two-night products from N+1 to 2N-1
num_product = num_nights*2
conf["num_product"] = num_product
product_resource_map = np.zeros((num_product, num_nights))
for i in range(1,num_nights):
    product_resource_map[i][i-1] = 1.0
    product_resource_map[i][i] = 1.0
for i in range(0,num_nights):    
    product_resource_map[i+num_nights][i] = 1.0
#product_resource_map[num_product-1][num_nights-1] = 1.0    
conf["product_resource_map"] = product_resource_map

product_revenue = 1000*np.random.uniform(size=[num_product])
product_revenue[product_null] = 0
conf["product_revenue"] = product_revenue
#total demand
product_demand = np.random.uniform(size=[num_product])*capacity
product_demand[product_null]  = 0
conf["product_demand"] = product_demand

num_steps = int(np.sum(product_demand)/0.01)
conf["num_steps"] = num_steps

#arrival rate (including)
product_prob = np.divide(product_demand,num_steps)
product_prob[0] = 1.0 - np.sum(product_prob)
conf["product_prob"] = product_prob

#computational graph generation

#define a state (in batch) and a linear value function
batch_size = 16
conf["batch_size"] = batch_size
#LHS is the value function for current state at time t
#for each state, we need num_nights real value inputs for available
# inventory, and +1 for time
dim_state_space = num_nights+1
conf["dim_state_space"] = dim_state_space

#tensorflow model inputs (or really state space samples)
#V(s,t)
#try neural network model: input->hidden->output



#define linear approximation model
#model = error_model_linear()
model = error_model_simple_nn()
model.build()

num_batches = 11
conf["num_batches"] = num_batches

first_run = True

saver = tf.train.Saver()
   
with tf.Session() as sess:    
    sess.run(tf.global_variables_initializer())    
        
    for batch in range(num_batches):
        #generate data for LHS V(s,t)
        
        data_lhs, data_rhs_1, data_rhs_2, data_mask, lp_bound_lhs, lp_bound_rhs_1, lp_bound_rhs_2 = generate_batch()
        
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
            print("Before even training, check parameters end\n")
                
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
        if 1 and batch % 10 == 0:              
            print("batch = ", batch)
            model.read_loss(sess
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
            print("\n")
            
    save_path = saver.save(sess, fname_output_model) 

    print("validation for random samples:")    
    for vb in range(1):
        print("validation batch ", vb)
        data_lhs, data_rhs_1, data_rhs_2, data_mask, lp_bound_lhs, lp_bound_rhs_1, lp_bound_rhs_2 = generate_batch()
        model.read_loss(sess
                    , data_lhs
                    , data_rhs_1
                    , data_rhs_2
                    , data_mask
                    , lp_bound_lhs
                    , lp_bound_rhs_1
                    , lp_bound_rhs_2)
        
    print("validation for monotonicity when state is fixed:")    
    for vb in range(1):
        print("validation batch ", vb)
        #data_lhs, data_rhs_1, data_rhs_2, data_mask, lp_bound_lhs, lp_bound_rhs_1, lp_bound_rhs_2 = generate_batch_t0()
        data_lhs, data_rhs_1, data_rhs_2, data_mask, lp_bound_lhs, lp_bound_rhs_1, lp_bound_rhs_2 = generate_batch_fix_time()
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

    print("total model building time = %.2f seconds" % (time.time()-ts), " time per batch = %.2f sec"%((time.time()-ts)/num_batches))
    
    policy_list = ["fifo", "dnn"]
    policy = dict(zip(policy_list,[policy_fifo(), policy_dnn()]))

    # next part is validation
    ts = time.time()    
    simulation()
    print("validation time = %.2f seconds"% (time.time()-ts))
    
#generate LP-DP decomposition policy
# we may need to save data in file and read
# during simulation in trunk    
def lpdp(param):
    #do a lp and use dual to do dp for each night
    num_nights = param["num_nights"]
    capacity = param["capacity"]
    product_demand = param["product_demand"]
    _, dual = lp(np.full(num_nights, capacity)
                , product_demand
                , conf
                , True)
    print(dual)
    
    for night in range(num_nights):
        print("solve DP for night", night)
    
    return 1.0