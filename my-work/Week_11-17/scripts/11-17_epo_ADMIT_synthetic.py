import pandas as pd
import sys
sys.path.append("../../..")
import numpy as np
import torch
import datetime
from scipy.interpolate import interp1d

from src.causalpfn.causal_estimator import CausalEstimator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Main hyperparameter
#N_DISC_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
N_DISC_VALUES = [2]

## Discretization function
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

## Synthetic data generation
data_name = "ADMIT-synthetic"
# DGP from https://github.com/CausalTeam/ADMIT/blob/main
def sigmoid(t): return 1. / (1. + np.exp(-1. * t))
n, d = 2000, 6
X = np.random.uniform(-1, 1, (n, d))
x1 = X[:, 0]
x2 = X[:, 1]
x3 = X[:, 2]
x4 = X[:, 3]
x5 = X[:, 4]
x6 = X[:, 5]
# Generate treatment 
mu_1 =  (10. * np.sin(np.maximum(x1, np.maximum(x2, x3))) + np.maximum(x3, np.maximum(x4, x5))**3)/(1. + (x1 + x5)**2) + np.sin(0.5 * x3) * (1. + np.exp(x4 - 0.5 * x3)) + x3**2 + 2. * np.sin(x4) + 2.*x5 - 6.5
T = np.random.normal(mu_1, 0.5)
T = sigmoid(T)
# Generate outcomes
mu_2 = np.cos(2 * np.pi * (T - 0.5)) * (T ** 2 + (4 * np.maximum(x1, x6) ** 3) * np.sin(x4) / (1 + 2 * x3 ** 2))
Y = np.random.normal(mu_2, 0.5)
# Generated expected potential outcomes empirically
treatment_doses = np.linspace(0, 1, 100)
effects = []
for t in treatment_doses:
    T_intervention = t * np.ones(T.shape)
    mu_2_intervention = np.cos(2 * np.pi * (T_intervention - 0.5)) * (T_intervention ** 2 + (4 * np.maximum(x1, x6) ** 3) * np.sin(x4) / (1 + 2 * x3 ** 2))
    effects.append(mu_2_intervention.mean())
effects = np.array(effects)

## Main inference loop
list_of_epos = [] # [(N_DISC, epos)], epos = [(mu_t0, mu_t1), (mu_t1, mu_t2), ... ]
for N_DISC in N_DISC_VALUES:
    print(f"N_DISC: {N_DISC}")
    discrete_treatment_levels = np.linspace(0, 1, N_DISC)
    T_discrete = discretize_treatment(T, N_DISC)
    epos = []
    for i, t in enumerate(discrete_treatment_levels[:-1]):
        t0, t1 = discrete_treatment_levels[i], discrete_treatment_levels[i + 1]
        ids = (np.abs(T_discrete - t0) < 1e-4) | (np.abs(T_discrete - t1) < 1e-4)
        T_temp = np.where(np.abs(T_discrete[ids] - t0) < 1e-4, 0, 1).astype(np.float32)
        X_temp = X[ids].astype(np.float32)
        Y_temp = Y[ids].astype(np.float32)
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

## Create DataFrame and format it
# treatment_value_idx refers to which bin of the N_DISC the data is in. 
# E.g. N_DISC = 3 and treatment_value_idx = 1 (out of bins [0, 1, 2])
# refers to a treatment value of 0.5, and N_DISC = 4 and treatment_value_idx = 2
# refers to a treatment value of 2/3. 
multi_indices = pd.MultiIndex.from_tuples(
    [(N, i) for N in N_DISC_VALUES for i in range(N)],
    names=["N_DISC", "treatment_value_idx"]
)
cols = ["EPOs_1", "EPOs_2", "true_effect"]
data = []
for i, N in enumerate(N_DISC_VALUES):
    epos = list_of_epos[i][1]
    for j in range(N): 
        # iterate over treatment_value_idx values; first and last values 
        # have only one prediction
        if j == 0:
            epo_1 = np.nan
            epo_2 = epos[j][0]
        elif j == N - 1:
            epo_1 = epos[j - 1][1]
            epo_2 = np.nan
        else:
            epo_1 = epos[j - 1][1]
            epo_2 = epos[j][0]
        treatment_val = j / (N - 1)
        true_effect = drf(treatment_val)
        data.append([epo_1, epo_2, true_effect])
epo_df= pd.DataFrame(
    data=data,
    index=multi_indices,
    columns=cols
)

## Saving output
# For filenaming so as not to overwrite previous output 
month = datetime.datetime.now().month
date = datetime.datetime.now().day
hour = datetime.datetime.now().hour
minute = datetime.datetime.now().minute
date_string = f"{month}-{date}_{hour}h{minute}m"
# Save the DataFrame
file_name = "EPO_df_" + data_name + "_" + date_string
file_location = "../output"
epo_df.to_csv(f"{file_location}/{file_name}.csv")
