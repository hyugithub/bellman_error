# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 13:53:28 2018

@author: hyu
"""

from functional_approximator import generate_batch, sample_generation
import config
import time
import numpy as np
import multiprocessing

def worker(fname, num_batch_gen, seed):
    conf2 = dict()
    config.param_init(conf2)
    np.random.seed(seed)
    sg = sample_generation(conf2)
    #batch generation return 7 outputs
    result = []
    for batch in range(num_batch_gen):
        result.append(generate_batch(conf2, sg))
#        result[0].append(data_lhs)
#        result[1].append(data_rhs_1)
#        result[2].append(data_rhs_2)
#        result[3].append(data_mask)
#        result[4].append(lp_bound_lhs)
#        result[5].append(lp_bound_rhs_1)
#        result[6].append(lp_bound_rhs_2)
        #result.append(generate_batch(conf2, sg))
    
    # shape of output is batch x batch_size x inherent dimension    
    data_lhs = np.stack([e[0] for e in result])
    data_rhs_1 = np.stack([e[1] for e in result])
    data_rhs_2 = np.stack([e[2] for e in result])
    data_mask = np.stack([e[3] for e in result])
    lp_bound_lhs = np.stack([e[4] for e in result])
    lp_bound_rhs_1 = np.stack([e[5] for e in result])
    lp_bound_rhs_2 = np.stack([e[6] for e in result])
    
    print(data_lhs.shape)
    print(data_rhs_1.shape)
    print(data_rhs_2.shape)
    print(data_mask.shape)
    print(lp_bound_lhs.shape)
    print(lp_bound_rhs_1.shape)
    print(lp_bound_rhs_2.shape)
    
    np.savez(fname
             , data_lhs=data_lhs
             , data_rhs_1=data_rhs_1
             , data_rhs_2=data_rhs_2
             , data_mask=data_mask
             , lp_bound_lhs=lp_bound_lhs
             , lp_bound_rhs_1=lp_bound_rhs_1
             , lp_bound_rhs_2=lp_bound_rhs_2            
             )
    
    return 0

def batch_data_prep():    
    
    num_processes = 2
    
    num_batch_gen = 3
    
    conf = dict()
    config.param_init(conf)
    np.random.seed(54321678)
    
    jobs = []            
    for p in range(num_processes):            
        fname_output = "".join(["batch_gen.", str(p), ".npz"])    
        s = np.random.choice(12345678)
        proc = multiprocessing.Process(target=worker
                                       , args=(fname_output
                                               , num_batch_gen
                                               , int(s)))
        jobs.append(proc)
    for j in jobs:
        j.start()
    for j in jobs:
        j.join()
    #    sg = sample_generation(conf)
    #    result = []
    #    for batch in range(num_batch_gen):
    #        result.append(generate_batch(conf, sg))
    
    result = [None] * num_processes
    for p in range(num_processes):
        fname_output = "".join(["batch_gen.", str(p), ".npz"])    
        result[p] = np.load(fname_output)
    
    data_lhs = np.stack([result[p]["data_lhs"] for p in range(num_processes)])
    data_rhs_1 = np.stack([result[p]["data_rhs_1"] for p in range(num_processes)])
    data_rhs_2 = np.stack([result[p]["data_rhs_2"] for p in range(num_processes)])
    data_mask = np.stack([result[p]["data_mask"] for p in range(num_processes)])
    lp_bound_lhs = np.stack([result[p]["lp_bound_lhs"] for p in range(num_processes)])
    lp_bound_rhs_1 = np.stack([result[p]["lp_bound_rhs_1"] for p in range(num_processes)])
    lp_bound_rhs_2 = np.stack([result[p]["lp_bound_rhs_2"] for p in range(num_processes)])     
    
    data_lhs = np.reshape(data_lhs, tuple([-1]+list(data_lhs.shape[2:])))
    data_rhs_1 = np.reshape(data_rhs_1, tuple([-1]+list(data_rhs_1.shape[2:])))
    data_rhs_2 = np.reshape(data_rhs_2, tuple([-1]+list(data_rhs_2.shape[2:])))
    data_mask = np.reshape(data_mask, tuple([-1]+list(data_mask.shape[2:])))
    lp_bound_lhs = np.reshape(lp_bound_lhs, tuple([-1]+list(lp_bound_lhs.shape[2:])))
    lp_bound_rhs_1 = np.reshape(lp_bound_rhs_1, tuple([-1]+list(lp_bound_rhs_1.shape[2:])))
    lp_bound_rhs_2 = np.reshape(lp_bound_rhs_2, tuple([-1]+list(lp_bound_rhs_2.shape[2:])))
    
    fname = "batch.npz"
    np.savez(fname
             , data_lhs=data_lhs
             , data_rhs_1=data_rhs_1
             , data_rhs_2=data_rhs_2
             , data_mask=data_mask
             , lp_bound_lhs=lp_bound_lhs
             , lp_bound_rhs_1=lp_bound_rhs_1
             , lp_bound_rhs_2=lp_bound_rhs_2            
             )
                       
    
    #EOF
def batch_data_load():
    fname = "batch.npz"
    data = np.load(fname)
    data_lhs = data["data_lhs"]
    data_rhs_1 = data["data_rhs_1"]
    data_rhs_2 = data["data_rhs_2"]
    data_mask = data["data_mask"]
    lp_bound_lhs = data["lp_bound_lhs"]
    lp_bound_rhs_1 = data["lp_bound_rhs_1"] 
    lp_bound_rhs_2 = data["lp_bound_rhs_2"]  
    
    
    print(data_lhs.shape)
    print(data_rhs_1.shape)
    print(data_rhs_2.shape)
    print(data_mask.shape)
    print(lp_bound_lhs.shape)
    print(lp_bound_rhs_1.shape)
    print(lp_bound_rhs_2.shape)
    
    return data_lhs, data_rhs_1, data_rhs_2 , data_mask, \
        lp_bound_lhs , lp_bound_rhs_1 , lp_bound_rhs_2
    #EOF

# sample for writing and reading
#if __name__ == '__main__':
#    for spyder    
#    __spec__ = None
#    ts = time.time()
#    batch_data_prep()
#    print("total batch data preparation time = %.2f"%(time.time()-ts))
#    batch_data_load()
    