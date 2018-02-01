# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 23:45:18 2018

@author: hyu
"""

#the purpose of this file is to simulate and validate
#performance of algorithm

import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import time
#from ortools.linear_solver import pywraplp
import itertools
from config import param_init
from lp_module import lp
from functional_approximator import *

class policy_fifo():      
    def do(self, s, r, p, tstep, param):
        #s is remaining capacity batch_size x state
        #r is resource needed for product p batch_size x ?night
        #p is products batch_size x 1
        #print(type(s), type(r), type(p))
        return (1.0-np.any((s-r)<0, axis=1)).astype(int)
    def close(self):
        return
    #EOC
    
class policy_lpdp():      
    def __init__(self, param):
        
        #fname = "../bid_price/value_function_lpdp.npy"
        
        self.num_steps = param["num_steps"]
        self.num_product = param["num_product"]
        self.batch_size = param["batch_size"]
        self.num_nights = param["num_nights"]
        self.product_resource_map = param["product_resource_map"]
        
        #from night x tstep x inventory to
        # tstep x night x inventory
        fname_policy = param["fname_policy"] 
        value = np.swapaxes(np.load(fname_policy), 0, 1)
        #bid price is delta V
        self.bid_price = value[:,:,1:] - value[:,:,:-1]
        
    def do(self, s, r, p, tstep, param):
        #print(type(s), type(r), type(p))
        product_revenue = param["product_revenue"]
        flag = (1.0-np.any((s-r)<0, axis=1)).astype(int)
        #return flag
        #batch_size = param["batch_size"]
        flag2 = np.zeros(self.batch_size)
        for b in range(self.batch_size):
            #product
            state = s[b]
            prod = p[b]
            rev = product_revenue[prod]
            bid_price = 0.0
            for night in range(self.num_nights):
                if self.product_resource_map[prod][night] >= 1e-6:
                    idx = max(state[night].astype(int)-1,0)
                    bid_price += self.bid_price[tstep][night][idx]
            if rev >= bid_price:
                flag2[b] = 1.0
        return np.logical_and(flag, flag2)
    def close(self):
        return
    #EOC    

class policy_lp_bound():                  
    def do(self, s, r, p, tstep, param):
        batch_size = param["batch_size"]
        product_null = param["product_null"]
        product_prob = param["product_prob"]
        capacity = param["capacity"]
        num_steps = param["num_steps"]
        #model = param["model"]
        #sess = param["sess"]
        product_revenue = param["product_revenue"]
        
        # check resource
        #print(s.shape, p.shape)
        flag = (1.0-np.any((s-r)<0, axis=1)).astype(int)

        #batch preparation -- avoid LP if possible
        lpb_lhs = np.ones(batch_size)
        lpb_rhs = np.ones(batch_size)
        for b, avail, prod in zip(range(batch_size),flag,p):
            if avail <= 1e-6 or prod == product_null:
                continue
            #otherwise it is not a null product and there is avail
            # so we must calculate Vs
            #please note that, because of the structure of our design
            #we have to generate lp bounds for all products, not only
            #the product currently being asssessed            
            
            lpb_lhs[b] = lp(s[b].astype(np.float32)
                        , (tstep*product_prob).astype(np.float32)
                        , param
                        )
            
            lpb_rhs[b] = lp((s[b]-r[b]).astype(np.float32)
                        , (tstep*product_prob).astype(np.float32)
                        , param
                        )                  
        #print(p.shape)
        bid_price = np.reshape(np.array([product_revenue[pp] for pp in p]), [batch_size,1])
        
        bid_price_delta = bid_price - np.reshape((lpb_lhs - lpb_rhs), (batch_size,-1))
        avail = (bid_price_delta >= 0.0).astype(int).flatten()
        
        #print(avail.shape, flag.shape)        
        return np.multiply(avail, flag)
    def close(self):
        return
    #EOC    
    
class policy_dnn():      
    def __init__(self, param):            
        param_init(param)           
        ts = time.time()    
        # we will need this later
        #if 0:
        tf.reset_default_graph()        
        #build model with variables to be initialized
        model = error_model_simple_nn(param)
        model.build()      
        self.model = model
        
        saver = tf.train.Saver()
        
        sess = tf.Session() 
        self.sess = sess
        fname_model = "C:/Users/hyu/Desktop/bellman/model/dpdnn.feb-01-2018_16_45_40.ckpt"
        # load tensorflow saved model
        saver.restore(sess, fname_model)        
        print("Tensorflow model building time = %.2f"%(time.time()-ts))        
    def do(self, s, r, p, tstep, param):
        #load parametrs
        batch_size = param["batch_size"]
        product_null = param["product_null"]
        product_prob = param["product_prob"]
        capacity = param["capacity"]
        num_steps = param["num_steps"]
        #model = param["model"]
        #sess = param["sess"]
        model = self.model
        sess = self.sess
        product_revenue = param["product_revenue"]
        
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
                        , param
                        )
            
            lpb_rhs[b] = lp((s[b]-r[b]).astype(np.float32)
                        , (tstep*product_prob).astype(np.float32)
                        , param
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
        
        #return flag
        #EOF
    def close(self):
        self.sess.close()
        
    #EOC        

def simulation():    
    conf = dict()
    param_init(conf)    
        
    # we should use the same seed for both fifo and new policy 
    # to reduce variance
    #load parameters 
    seed_simulation = conf["seed_simulation"]  
    np.random.seed(seed_simulation)
    
    #generate seeds for each batch demand    
    
    #policy_list = conf["policy_list"]
    policy_list = ["fifo", "dnn", "lpdp", "lp_bound"]
    #policy_list = ["fifo", "dnn"]
    
    #policy = dict(zip(policy_list,[policy_fifo(), policy_dnn()]))
            
    #30 x 64 gives us about 2000 samples 
    # as part of validation. we think the sample 
    # mean should be good. not sure about variance
    #num_iterations = 30
    num_iterations = 1
    
    seed_demand = np.random.choice(32452843, [num_iterations]) % 15485863	
    
    # initial state
    #state_initial = np.ones([batch_size, num_nights])*capacity
 
    for i in range(num_iterations): 
        
        seed_dem = seed_demand[i]

        
        #revenue["fifo"] = np.zeros(batch_size)
        #revenue["dnn"]  = np.zeros(batch_size)
        
        for p in policy_list:
            if p == "fifo":
                policy = policy_fifo()
            if p == "dnn":
                policy = policy_dnn(conf)
            if p == "lpdp":
                #policy = policy_fifo()
                policy = policy_lpdp(conf)            
            if p == "lp_bound":
                policy = policy_lp_bound()            
            
            batch_size = conf["batch_size"]
            num_product = conf["num_product"]
            num_steps = conf["num_steps"]
            product_prob = conf["product_prob"]
            num_nights = conf["num_nights"]
            capacity = conf["capacity"]
            product_resource_map = conf["product_resource_map"]
            product_revenue = conf["product_revenue"]   
            revenue = np.zeros(batch_size)
            # for each time step, generate demand
            np.random.seed(seed_dem)
            demand = np.random.choice(range(num_product)
                                        , size=(num_steps, batch_size)
                                        , p=product_prob
                                     )
            state = np.ones([batch_size, num_nights])*capacity
            
            for s in np.arange(start=num_steps-1, stop=0, step=-1):
                resource = np.stack([product_resource_map[p] for p in demand[s]])
                
                admit = policy.do(state, resource, demand[s], s, conf)
                revenue0 = np.array([product_revenue[p]*admit[b] for b,p in zip(range(batch_size), demand[s])])
                revenue += revenue0
                state = state - np.multiply(resource, np.reshape(admit, [batch_size,1]))
#        for r1,r2,r3,r4 in zip(revenue["fifo"], revenue["dnn"], revenue["lpdp"], revenue["lp_bound"]):
#            print("dnn lift = %.2f"%(r2/r1-1.0)
#                , "lp bound lift = %.2f"%(r4/r1-1.0)
#                , "lpdp lift = %.2f"%(r3/r1-1.0)
#                )      
            policy.close()
#        for r1,r2,in zip(revenue["fifo"], revenue["dnn"]):
#            print("dnn lift = %.2f"%(r2/r1-1.0))
        
#        for pol in policy_list:
#            print("policy ", pol, " revenue = {:,}".format(np.mean(revenue[pol])))
            
if __name__ == '__main__':    
#    for spyder    
    __spec__ = None    
    ts = time.time()
    simulation()
    print("total batch data preparation time = %.2f"%(time.time()-ts))
