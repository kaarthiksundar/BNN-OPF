import numpy as np
from dc3classes  import ProblemData 
from dc3feasibility import assess_feasibility
from sklearn.model_selection import train_test_split   # just for convenience

# Suppose X.shape == (N, m)   and   Y.shape == (N, n)
#filename = 'random_nonconvex_dataset_var100_ineq50_eq50_ex10000.npz'
filename = "random_nonconvex_dataset_var20_ineq5_eq10_ex5000.npz"
#filename = 'random_nonconvex_dataset_var150_ineq50_eq50_ex5000.npz'
data = np.load(filename, allow_pickle=False)
G, Q, A, h, p, X, Y = (data[k] for k in ('G','Q','A','h','p','X','Y'))
X_u, X_val, Y_u, Y_val = train_test_split(X, Y, test_size=0.1, random_state=0)

eps = 1e-6  # avoids division‑by‑zero

# ── inputs ─────────────────────────────────────────────
X_mean = X_u.mean(axis=0)           # (m,)
X_std  = X_u.std(axis=0) + eps      # (m,)
X_u_norm  = (X_u  - X_mean) / X_std
X_val_norm = (X_val - X_mean) / X_std

# ── outputs (optional but recommended) ─────────────────
Y_mean = Y_u.mean(axis=0)           # (n,)
Y_std  = Y_u.std(axis=0) + eps      # (n,)

pdata = ProblemData(
    Q=Q, A=A, G=G, p=p, h=h,
    X_train_norm=X_u_norm, X_train=X_u,
    Y_train = Y_u, Y_val = Y_val,
    X_val_norm=X_val_norm, X_val=X_val,
    Y_mean=Y_mean, Y_std=Y_std
)
pdata.init_meta(hidden_width=256, num_hidden_layers=3)

err = assess_feasibility(X,Y,pdata)
print(err)
