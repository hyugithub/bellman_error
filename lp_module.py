import numpy as np
import time
from ortools.linear_solver import pywraplp
import itertools

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