import pandas as pd
from mip import Model, xsum, minimize, BINARY, INTEGER, CONTINUOUS, CBC, GRB, OptimizationStatus
from vrp_utils import get_dist_mat
import matplotlib.pyplot as plt

# constants
num_vehicle = 25
capacity = 200

data = pd.read_csv("instance.csv")
data = data.loc[0:25]
origin = data.loc[0:0]
data = pd.concat([data, origin])
demand = data["DEMAND"].tolist()
service_time = data["SERVICE_TIME"].tolist()
coord = list(zip(data["XCOORD"], data["YCOORD"]))
n = len(coord) - 2

dist_mat = get_dist_mat(coord)
a = data["READY_TIME"].tolist()
b = data["DUE_DATE"].tolist()
big_m = 1e4
big_m2 = n + 5
delta = 1e-3
lmd = 1e3
# sets
N = range(n + 2)
C = range(1, n + 1)
V = range(num_vehicle)

# modelling
m = Model(solver_name=GRB)

x = {(i, j, k): m.add_var(var_type=BINARY, name=f"x_{i}_{j}_{k}") for i in N for j in N for k in V}
s = {(i, k): m.add_var(var_type=CONTINUOUS, name=f"s_{i}_{k}") for i in N for k in V}
y = {k: m.add_var(var_type=BINARY, name=f"y_{k}") for k in V}

objective = xsum(dist_mat[(i, j)] * x[(i, j, k)] for i in N for j in N for k in V) + lmd * xsum(y[k] for k in V)

m.objective = minimize(objective)

# prunes
for i in N:
    for k in V:
        j = i
        m.add_constr(x[(i, j, k)] == 0, name=f"prune1_{i}_{j}_{k}")

for j in N:
    for k in V:
        m.add_constr(x[(n + 1, j, k)] == 0, name=f"prune2_{n + 1}_{j}_{k}")

for i in N:
    for k in V:
        m.add_constr(x[(i, 0, k)] == 0, name=f"prune3_{i}_{0}_{k}")

# each customer is visited once
for i in C:
    m.add_constr(xsum(x[(i, j, k)] for j in N for k in V) == 1, name=f"visit_{i}")

# capacity constraint
for k in V:
    m.add_constr(xsum(demand[i] * xsum(x[(i, j, k)] for j in N) for i in C) <= capacity, name=f"capacity_{k}")

# arc balancing
for h in C:
    for k in range(num_vehicle):
        m.add_constr(xsum(x[(i, h, k)] for i in N) == xsum(x[(h, j, k)] for j in N), name=f"arc_{h}_{k}")

# time window constraints
for i in N:
    for j in N:
        for k in V:
            m.add_constr(s[i, k] + service_time[i] + dist_mat[(i, j)] <= s[j, k] + big_m * (1 - x[(i, j, k)]),
                         name=f"tw1_{i}_{j}_{k}")

for i in N:
    for k in V:
        m.add_constr(s[i, k] >= a[i], name=f"tw2_{i}_{k}")
        m.add_constr(s[i, k] <= b[i], name=f"tw3_{i}_{k}")

# vehicle usage
for k in V:
    m.add_constr(xsum(x[i, j, k] for i in N for j in N) >= 1 + delta - big_m2 * (1 - y[k]), name=f"vu1_{k}")
    m.add_constr(xsum(x[i, j, k] for i in N for j in N) <= 1 + big_m2 * y[k], name=f"vu2_{k}")

# each vehicle starts from the depot
for k in range(num_vehicle):
    m.add_constr(xsum(x[(0, j, k)] for j in N) == y[k], name=f"start_{k}")

# each vehicle ends at the depot
for k in range(num_vehicle):
    m.add_constr(xsum(x[(i, n + 1, k)] for i in N) == y[k], name=f"end_{k}")
m.write("model.lp")
m.optimize()

# parse solution
y_list = [yi.x for yi in y.values()]

x_list = [idx for idx, xi in x.items() if xi.x > 0.9]
