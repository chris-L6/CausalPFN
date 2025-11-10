import pandas as pd
import sys
sys.path.append("..")
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.causalpfn import ATEEstimator
from functools import reduce
from debt_loss_dgp import DebtDGP, TREATMENT, OUTCOME

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## MAIN HYPERPARAMETER
# N_DISC = 17, 19 were seen to be the worst, so I won't go higher than 16
N_DISC_VALUES = [10]

# Load data
generator = DebtDGP()
n = 15000
df = generator.generate_random_financial_data(n)
generator.fit_scalers(df)
df[TREATMENT] = generator.generate_treatment(df, noise=True, noise_scale=0.0, treament_noise_std=0.07)
df[OUTCOME + "_probs"], df[OUTCOME] = generator.calculate_outcome_probability(df, df[TREATMENT])

# Causal curves
step = 1e-2
bin_edges = np.arange(0, 1+step, step)
real_dose_response = generator.compute_causal_effects(df, bin_edges, "ate")
bin_edges_contained = bin_edges.copy()
bin_edges_contained[0] = -1
df['treatment_bin'] = pd.cut(df[TREATMENT], bins=bin_edges_contained, labels=False)
mean_outcome = [c[0] for c in df.groupby(['treatment_bin'])[[OUTCOME]].mean().to_numpy().tolist()]

# Plotting
fig = plt.figure()
plt.plot(bin_edges, real_dose_response, label="Real dose-response curve")
plt.plot(bin_edges[1:], mean_outcome, label="Observed mean Outcome")

plt.plot(df[TREATMENT], df[OUTCOME + "_probs"], '.', alpha=0.2)
plt.legend(loc="upper left")
plt.xlabel("Percentage of debt loss (%)")
plt.ylabel("Probability of debt repayment")
plt.title("Simpson's Paradox")
plt.savefig("../debt_dgp_fig_1.png")

# Preprocessing
X = df[generator.confounders].values.astype(np.float32)
T = df[TREATMENT].values.astype(np.float32)
Y = df[OUTCOME + "_probs"].values.astype(np.float32)

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

## Main prediction loop. Takes about 6 minutes to run.
list_of_ates = [] # [(N_DISC, ates)]

for N_DISC in N_DISC_VALUES:
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
            verbose=True
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
y_true = np.interp(t_mesh, np.arange(0, 1+step, step), real_dose_response)

# Get RMSE by evaluating on each of the t_mesh points
rmse_dict = dict()
for N_DISC, ates in list_of_ates:
    epos = [sum(ates[:i]) for i in range(N_DISC)]
    y_pred = np.interp(t_mesh, np.linspace(0, 1, N_DISC), epos)
    error = calculate_rmse(y_pred, y_true)
    rmse_dict[N_DISC] = np.round(error, 4)

# Create dataframe
df = pd.DataFrame.from_dict(rmse_dict, orient="index", columns=["RMSE"])
df.to_csv("../output/rmse_table.csv")
fig = plt.figure()
plt.bar(df.index, height=df["RMSE"])
plt.xlabel("N_DISC")
plt.ylabel("RMSE")
plt.title("RMSE of different choices of N_DISC")
plt.savefig("../output/rmse_bar_chart.png")

# Plot predictions
fig = plt.figure(figsize=(15, 10))
i = 0 # counter for visual effects
for N_DISC, ates in list_of_ates:
    epos = [sum(ates[:i]) for i in range(N_DISC)]
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
