## Module imports
import pandas as pd
import sys
sys.path.append("..")
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.causalpfn import ATEEstimator
from functools import reduce

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


## MAIN HYPERPARAMETER
N_DISC_VALUES = [2, 4, 5]

## Main body
# Discretization function
def discretize_treatment(T: np.ndarray, N: int) -> np.ndarray:
    """Returns discretized version of T. Assumes range of T is [0, 1].

    Args:
        T (np.ndarray): The raw treatment data 
        N (int): The number of discretized treatment values to use

    Returns:
        np.ndarray: The discretized treatment data
    """
    dt = 1 / (N - 1)
    T_discrete = np.round(np.array(T, dtype=np.float32) / dt) * dt

    return T_discrete

# RMSE function
def calculate_rmse(y_pred, y_true):
    result = np.mean(np.power(y_pred - y_true, 2))
    result = np.sqrt(result)

    return result


## Synthetic data generation
# Generate synthetic data, using the DGP from Vahid's paper - nonlinear backdoor
np.random.seed(42)
n, d = 2000, 3
X = np.random.normal(1, 1, size=(n, d)).astype(np.float32)
T = (X[:, 0] - X[:, 1] + 2 * X[:, 2] + 2 + np.random.normal(0, 3, size=n)).astype(np.float32)
T = T - T.min() # Rescale
T = T / T.max() # Rescale
Y = (3 * X[:, 0] + X[:, 1] - 0.5 * X[:, 2] + 3 * T + np.random.normal(0, 2, size=n)).astype(np.float32)
def drf(t): return 3.5 + 3 * t # true dose-response function

df = pd.concat([
    pd.DataFrame(data=X, columns=["x1", "x2", "x3"]), 
    pd.DataFrame(data=T, columns=["T"]), 
    pd.DataFrame(data=Y, columns=["Y"])
    ], axis=1)

## Main prediction loop. Takes about 6 minutes to run.
list_of_ates = [] # [(N_DISC, ates)]

for N_DISC in N_DISC_VALUES:
    print(f"N_DISC: {N_DISC}")
    discrete_treatment_levels = np.linspace(0, 1, N_DISC)
    T_discrete = discretize_treatment(T, N_DISC)
    ates = []
    for i, t in enumerate(discrete_treatment_levels[:-1]):
        t0, t1 = discrete_treatment_levels[i], discrete_treatment_levels[i + 1]
        ids = (np.abs(T_discrete - t0) < 1e-4) | (np.abs(T_discrete - t1) < 1e-4)
        T_temp = np.where(np.abs(T_discrete[ids] - t0) < 1e-4, 0, 1).astype(np.float32)
        X_temp = X[ids].astype(np.float32)
        Y_temp = Y[ids].astype(np.float32)
        causalpfn_ate = ATEEstimator(
            device=device,
            verbose=False
        )
        causalpfn_ate.fit(X_temp, T_temp, Y_temp)
        ate = causalpfn_ate.estimate_ate()
        ates.append(ate)
    list_of_ates.append((N_DISC, ates))

# To get a good RMSE estimate for all the predictions, need a large mesh
fine_mesh = np.linspace(0, 1, 100) # fixed, fine mesh
t_mesh = [np.linspace(0, 1, N_DISC) for N_DISC in N_DISC_VALUES] # to ensure all discrete treatment values are hit
t_mesh += [fine_mesh]
t_mesh = reduce(np.union1d, tuple(t_mesh))

# Get RMSE by evaluating on each of the t_mesh points
rmse_dict = dict()
y_true = drf(t_mesh)
for N_DISC, ates in list_of_ates:
    epos = [3.5 + sum(ates[:i]) for i in range(N_DISC)]
    y_pred = np.interp(t_mesh, np.linspace(0, 1, N_DISC), epos)
    error = calculate_rmse(y_pred, y_true)
    rmse_dict[N_DISC] = np.round(error, 4)

# Create dataframe, print optimizer, and save the data
df = pd.DataFrame.from_dict(rmse_dict, orient="index", columns=["RMSE"])
row = df[df["RMSE"] == df["RMSE"].min()]
print(f"Optimal value of N_DISC: {row.index.values[0]} (RMSE: {row.values.item():.4f})")
df.to_csv("../output/rmse-values-linear-backdoor.csv")

fig = plt.figure()
plt.bar(df.index, height=df["RMSE"])
plt.xlabel("N_DISC")
plt.ylabel("RMSE")
plt.title("RMSE of different choices of N_DISC")
plt.savefig("../output/rmse_bar_chart-linear-backdoor.png")

# Plot predictions
fig = plt.figure(figsize=(15, 10))
i = 0 # counter for visual effects
for N_DISC, ates in list_of_ates:
    epos = [3.5 + sum(ates[:i]) for i in range(N_DISC)]
    if df.loc[N_DISC, "RMSE"] == df.min().values: # Emphasize best prediction
        plt.plot(np.linspace(0.0, 1.0, N_DISC), epos, '-', linewidth=2,
                 label=f"N_DISC={N_DISC} (Best)", c="blue", zorder=14)
    else:
        plt.plot(np.linspace(0.0, 1.0, N_DISC), epos, '--', linewidth=1,
                 label=f"N_DISC={N_DISC}")
    i += 1

# True dose-response curve
plt.plot(np.linspace(0.0, 1.0, len(y_true)), y_true,
         label="True dose-response curve", linewidth=2, c='k', zorder=15)

plt.title("$E[Y_t]$ (CausalPFN estimate using ATEs)") # Note E[Y_t] = E[Y | do(T = t)]
plt.xlabel("Treatment dosage t")
plt.ylabel("Expected outcome Y")

plt.legend(fontsize=8)
plt.savefig("../output/drf_plot.png", dpi=500)
