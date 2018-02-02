from multiprocessing import Pool
import numpy as np

def f(xin):
    x,y = xin
    s = 0.0
    for k in range(12345678):
        s += np.log(k+1)
    print(y)
    return x*x

if __name__ == '__main__':
    args = range(10)
    args = zip(args, [str(x) for x in args])
    with Pool(5) as p:
        print(p.map(f, args))
