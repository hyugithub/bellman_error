# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 13:53:28 2018

@author: hyu
"""

from functional_approximator import generate_batch, sample_generation
import config
import time

ts = time.time()

num_processes = 2

num_batch_gen = 10

conf = dict()
config.param_init(conf)

for p in range(num_processes):
    sg = sample_generation(conf)
    for batch in range(num_batch_gen):
        data_lhs, data_rhs_1, data_rhs_2, data_mask, lp_bound_lhs, lp_bound_rhs_1, lp_bound_rhs_2 = generate_batch(conf, sg)
               
print("total batch data preparation time = %.2f"%(time.time()-ts)
    , "per batch = %.2f"% ((time.time()-ts)/(num_batch_gen*num_processes)))

