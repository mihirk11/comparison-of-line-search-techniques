import scipy as sp;
import numpy as np;
import scipy.optimize;
print  np.__version__
import pylab as pl
import pandas as pd
import matplotlib.pyplot as plt
def test_func(x):
    return (x[0])**2+(x[1])**2

def test_grad(x):
    return [2*x[0],2*x[1]]



def test_func(x):
    return (x[0])**2+(x[1])**2

def test_func(x):
    return (x[0])**2
def test_grad(x):
    return 2*x[0]
def f(x):
    return (x - 2) * x * (x + 2)**2

def f2(x):
    return (x)**2



lst=[]
for x in range(1, 1000):
    res= sp.optimize.line_search(test_func,test_grad,np.array([x]),np.array([-10]))
    lst.append(res[1])
plt.plot(lst)
plt.show()


lst=[]
for x in range(1, 100):
    res= scipy.optimize.minimize(test_func,np.array([x]),method='Nelder-Mead')
    lst.append(res.nfev)
plt.plot(lst)
plt.show()


print scipy.optimize.minimize(test_func,np.array([1000]),method='BFGS')







lst= []
for x in range(1, 100000):
    res= sp.optimize.minimize_scalar(f2, bracket= (-x,x)  ,method='golden')
    lst.append(res.nfev)
    print x
print lst

plt.xscale('log')
plt.plot(lst)
plt.show()
res=sp.optimize.golden(f2,full_output=1)
print res

res= sp.optimize.minimize_scalar(f2,method='brent')
print res