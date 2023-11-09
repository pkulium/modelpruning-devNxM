import miosqp
import numpy as np
import scipy.sparse as sp

# Settings
miosqp_settings = {
    'verbose': True,
    'eps_int_feas': 1e-03,
    'max_iter_bb': 1000,
    'branching_rule': 0,
    'eps_rel_lb': 1e-03,
    'eps_abs_lb': 1e-03,
    'tree_explor_rule': 0,
    'tree_explor_depth': 1e3,
    'cutting_plane_alg': 0
}

osqp_settings = {
    'eps_abs': 1e-03,
    'eps_rel': 1e-03,
    'eps_prim_inf': 1e-04,
    'eps_dual_inf': 1e-04,
    'polish': True
}

# Problem setup
n = 20  # Dimension of the vector z
m = 4   # Block size
sparsity = 2  # Number of zeros in each block

# Example vector z
z = np.random.randn(n)

# Define P and q for the objective function
P = sp.eye(n).tocsc()  # Convert to csc format for miosqp
q = -z

# Define A, l, u for the constraints
A_list = []
l_list = []
u_list = []


# Add bounds on s
for i in range(n):
    row = sp.lil_matrix((1, n))
    row[0, i] = 1
    A_list.append(row)
    l_list.append(0)
    u_list.append(1)

# Convert lists to sparse matrices
A = sp.vstack(A_list).tocsc()  # Convert to csc format for miosqp
l = np.array(l_list)
u = np.array(u_list)

# Define integer constraints (no integer variables for now)
i_idx = np.array([], dtype=np.int_)
i_l = np.array([], dtype=np.float64)
i_u = np.array([], dtype=np.float64)


# Setup MIOSQP problem
model = miosqp.MIOSQP()
model.setup(P, q, A, l, u, i_idx, i_l, i_u, miosqp_settings, osqp_settings)

# Solve problem
results = model.solve()

# Retrieve the optimal vector s if the problem was solved
if results.status != 'Solved':
    print("The problem was not solved to optimality. Status: ", results.status)
else:
    s_optimal = results.x
    print("The optimal vector s is: ", s_optimal)
