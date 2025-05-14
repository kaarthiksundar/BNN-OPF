import numpy as np
import pickle
import torch
import torch.nn.functional as F
import numpy as np
import numpy.linalg as npl

import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], os.pardir, os.pardir))
from utils import NonconvexProblem

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#filename = 'random_nonconvex_dataset_var100_ineq50_eq50_ex10000'
#filename = 'random_nonconvex_dataset_var150_ineq50_eq50_ex5000'
#filename = "random_nonconvex_dataset_var20_ineq5_eq10_ex5000"
filename = 'random_nonconvex_dataset_var70_ineq50_eq20_ex5000'
with open(filename, 'rb') as file:
    data = pickle.load(file)

print(data)

Y = data.Y.detach().numpy()
X = data.X.detach().numpy()
Q = data.Q.detach().numpy()
A = data.A.detach().numpy()
G = data.G.detach().numpy()
h = data.h.detach().numpy()
p = data.p.detach().numpy()

T1 =  np.einsum("ij,kj->ki", A,Y)
print("Max Eq",npl.norm(T1 - X, ord = np.inf))
T2 = np.einsum("ij,kj->ki", G,Y)
print("Max InEq", npl.norm(np.maximum(0,T2 - h), ord = np.inf))

T3 =  0.5*np.einsum("ij,jk,ik -> i",Y,Q,Y)
T4 = np.einsum("i,ji->j",p, np.sin(Y))
Obj = T3 + T4

np.savez(filename + '.npz', G=G, Q = Q, A=A, h=h, p=p, X=X, Y=Y)   # or savez_compressed
