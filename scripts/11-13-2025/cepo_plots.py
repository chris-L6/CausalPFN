## The goal of this script is to reproduce the results of the nonlinear backdoor 
## (from Vahid's paper) notebook

## Module imports
import pandas as pd
import sys
sys.path.append("..")
import numpy as np
import torch
import matplotlib.pyplot as plt
import json

from src.causalpfn.causal_estimator import CausalEstimator
from src.causalpfn import ATEEstimator
from functools import reduce

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## MAIN HYPERPARAMETER
N_DISC_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10]

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
n, d = 2000, 1
X = np.random.normal(2, 1, size=(n, d)).astype(np.float32)
T = (0.1 * X[:, 0] ** 2 - X[:, 0] + np.random.normal(1, 2, size=n)).astype(np.float32)
T = T - T.min() # Rescale
T = T / T.max() # Rescale
Y = (0.5 * T ** 2 - T * X[:, 0] + np.random.normal(0, 2, size=n)).astype(np.float32)
def drf(t): return 0.5 * t ** 2 - 2 * t # true dose-response funcion

df = pd.concat([
    pd.DataFrame(data=X, columns=["x"]), 
    pd.DataFrame(data=T, columns=["T"]), 
    pd.DataFrame(data=Y, columns=["Y"])
    ], axis=1)

list_of_epos = [] # [(N_DISC, epos)], epos = [(mu_t0, mu_t1), (mu_t1, mu_t2), ... ]
list_of_ates = [] # [(N_DISC, ates)], ates = [ATE(t0, t1), ATE(t1, t2), ...]

for N_DISC in N_DISC_VALUES:
    print(f"N_DISC: {N_DISC}")
    discrete_treatment_levels = np.linspace(0, 1, N_DISC)
    T_discrete = discretize_treatment(T, N_DISC)
    epos = []
    ates = []
    for i, t in enumerate(discrete_treatment_levels[:-1]):
        t0, t1 = discrete_treatment_levels[i], discrete_treatment_levels[i + 1]
        ids = (np.abs(T_discrete - t0) < 1e-4) | (np.abs(T_discrete - t1) < 1e-4)
        T_temp = np.where(np.abs(T_discrete[ids] - t0) < 1e-4, 0, 1).astype(np.float32)
        X_temp = X[ids].astype(np.float32)
        Y_temp = Y[ids].astype(np.float32)
        # to predict ate
        causalpfn_ate = ATEEstimator(
            device=device,
            verbose=True
        )
        causalpfn_ate.fit(X_temp, T_temp, Y_temp)
        ate = causalpfn_ate.estimate_ate()
        ates.append(ate)
        # to predict cepo
        X_context = X_temp 
        t_context = T_temp
        y_context = Y_temp
        X_query = X_temp 
        t_all_ones = np.ones(X_query.shape[0], dtype=X_query.dtype)
        t_all_zeros = np.zeros(X_query.shape[0], dtype=X_query.dtype)
        causalpfn_cepo = CausalEstimator(
            device=device,
            verbose=True
        )
        causalpfn_cepo.fit(X_temp, T_temp, Y_temp)
        mu_vals = causalpfn_cepo._predict_cepo(
            X_context=X_context,
            t_context=t_context,
            y_context=y_context,
            X_query=np.concatenate([X_query, X_query], axis=0),
            t_query=np.concatenate([t_all_zeros, t_all_ones], axis=0),
            temperature=causalpfn_cepo.prediction_temperature,
        )
        mu_0 = (mu_vals[: X_query.shape[0]]).mean()
        mu_1 = (mu_vals[X_query.shape[0] :]).mean()
        epos.append((mu_0, mu_1))
    list_of_epos.append((N_DISC, epos))
    list_of_ates.append((N_DISC, ates))

# import epos and ates to json
with open("../../output/11-13-2025/epos.json", "w", encoding='utf-8') as f:
    json.dump(list_of_epos, f, ensure_ascii=False, indent=4)
with open("../../output/11-13-2025/ates.json", "w", encoding='utf-8') as f:
    json.dump(list_of_ates, f, ensure_ascii=False, indent=4)