# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 23:45:18 2018

@author: hyu
"""

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

class policy_fifo():      
    def do(self, s, r, p, tstep, param):
        #s is remaining capacity batch_size x state
        #r is resource needed for product p batch_size x ?
        #p is products batch_size x 1
        #print(type(s), type(r), type(p))
        return (1.0-np.any((s-r)<0, axis=1)).astype(int)
    #EOC
    
class policy_lpdp():      
    def __init__(self, param):
        fname = "../bid_price/value_function_lpdp.npy"
        
        self.num_steps = param["num_steps"]
        self.num_product = param["num_product"]
        self.batch_size = param["batch_size"]
        self.num_nights = param["num_nights"]
        self.product_resource_map = param["product_resource_map"]
        
        #from night x tstep x inventory to
        # tstep x night x inventory
        value = np.swapaxes(np.load(fname), 0, 1)
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
    #EOC    
    
class policy_dnn():      
    def do(self, s, r, p, tstep, param):
        #load parametrs
        batch_size = param["batch_size"]
        product_null = param["product_null"]
        product_prob = param["product_prob"]
        capacity = param["capacity"]
        num_steps = param["num_steps"]
        model = param["model"]
        sess = param["sess"]
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
        
    #EOC        

def simulation(param):    
    # we should use the same seed for both fifo and new policy 
    # to reduce variance
    #load parameters 
    seed_simulation = param["seed_simulation"]
    batch_size = param["batch_size"]
    num_product = param["num_product"]
    num_steps = param["num_steps"]
    product_prob = param["product_prob"]
    num_nights = param["num_nights"]
    capacity = param["capacity"]
    product_resource_map = param["product_resource_map"]
    product_revenue = param["product_revenue"]
    batch_size = param["batch_size"]
    
    
    #policy_list = param["policy_list"]
    policy_list = ["fifo", "dnn", "lpdp"]
    
    #policy = dict(zip(policy_list,[policy_fifo(), policy_dnn()]))
    
    policy = dict()
    for p in policy_list:
        if p == "fifo":
            policy[p] = policy_fifo()
        if p == "dnn":
            policy[p] = policy_dnn()
        if p == "lpdp":
            #policy[p] = policy_fifo()
            policy[p] = policy_lpdp(param)            
        
    np.random.seed(seed_simulation)
    # initial state
    #state_initial = np.ones([batch_size, num_nights])*capacity
    
    for _ in range(1):
        revenue = dict(zip(policy_list, [np.zeros(batch_size)]*len(policy_list)))
        #revenue["fifo"] = np.zeros(batch_size)
        #revenue["dnn"]  = np.zeros(batch_size)
        
        # for each time step, generate demand
        demand = np.random.choice(range(num_product)
                                    , size=(num_steps, batch_size)
                                    , p=product_prob
                                 )
        state = dict(zip(policy_list
                , [np.ones([batch_size, num_nights])*capacity]*len(policy_list)
                ))
        #state["fifo"] = np.ones([batch_size, num_nights])*capacity
        #state["dnn"] = np.ones([batch_size, num_nights])*capacity
            
        for s in np.arange(start=num_steps-1, stop=0, step=-1):
            resource = np.stack([product_resource_map[p] for p in demand[s]])
            for pol in policy_list:
                admit = policy[pol].do(state[pol], resource, demand[s], s, param)
                revenue0 = np.array([product_revenue[p]*admit[b] for b,p in zip(range(batch_size), demand[s])])
                revenue[pol] = revenue[pol] + revenue0
                state[pol] = state[pol] - np.multiply(resource, np.reshape(admit, [batch_size,1]))
        for r1,r2,r3 in zip(revenue["fifo"], revenue["dnn"], revenue["lpdp"]):
            print("dnn lift = %.2f"%(r2/r1-1.0)
                , "lpdp lift = %.2f"%(r3/r1-1.0)
                )                