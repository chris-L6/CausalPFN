import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import datetime
import sys
sys.path.append("../discretized-causalpfn")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from discretize import DataDiscretizer
from inference import DiscreteCausalPFN
from scipy.interpolate import interp1d

## HELPER FUNCTIONS AND PARAMETERS
# RMSE function
def calculate_rmse(y_pred, y_true):
    result = np.mean(np.power(y_pred - y_true, 2))
    result = np.sqrt(result)

    return result

fine_t_mesh = np.linspace(0, 1, 100)

# Parameters
scheme = "uniform"
comparison_method = "all"
N_DISC_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

## DGP 
data_name = "debt-loss"
generator = DebtDGP()
n = 15000
df = generator.generate_random_financial_data(n)
generator.fit_scalers(df)
df[TREATMENT] = generator.generate_treatment(df, noise=True, noise_scale=0.0, treament_noise_std=0.07)
df[OUTCOME + "_probs"], df[OUTCOME] = generator.calculate_outcome_probability(df, df[TREATMENT])
step = 1e-2
bin_edges = np.arange(0, 1+step, step)
real_dose_response = generator.compute_causal_effects(df, bin_edges, "ate")
bin_edges_contained = bin_edges.copy()
bin_edges_contained[0] = -1
df['treatment_bin'] = pd.cut(df[TREATMENT], bins=bin_edges_contained, labels=False)
mean_outcome = [c[0] for c in df.groupby(['treatment_bin'])[[OUTCOME]].mean().to_numpy().tolist()]
# True dose-response function
y_true_f = interp1d(bin_edges, real_dose_response)
def drf(t):
    return y_true_f(t)
X = df[generator.confounders].values.astype(np.float32)
T = df[TREATMENT].values.astype(np.float32)
Y = df[OUTCOME + "_probs"].values.astype(np.float32)

## INFERENCE
epos_collection = dict() # collect all results across all N_DISC in N_DISC_VALUES
CI_collection = dict() 
for N_DISC in N_DISC_VALUES:
    print(f"N_DISC: {N_DISC}")
    discretizer = DataDiscretizer(scheme=scheme)
    T_discrete, T_vals = discretizer.discretize_treatment(T, N_DISC)
    model = DiscreteCausalPFN(
        comparison_method=comparison_method,
        N_DISC=N_DISC,
        device=device,
        verbose=True
    )
    epos_dict, CI_dict = model.predict_epos(
        X,
        T_discrete, 
        Y, 
        discrete_treatment_vals=T_vals,
        take_mean=True, 
        alpha=0.5
    )
    epos_collection[N_DISC] = epos_dict
    CI_collection[N_DISC] = CI_dict

## CREATING AND EXPORTING DATA
multi_indices = pd.MultiIndex.from_tuples(
    [(N, t) for N in epos_collection for t in epos_collection[N]],
    names=["N_DISC", "T"]
)
cols = ["estimated_epo", "CI_lower_bound", "CI_upper_bound"]
epo_df = pd.DataFrame(index=multi_indices, 
                      columns=cols)
for N in epos_collection:
    for t in epos_collection[N]:
        epo_df.loc[(N, t), "estimated_epo"] = epos_collection[N][t]
        epo_df.loc[(N, t), "CI_lower_bound"] = CI_collection[N][t][0].item()
        epo_df.loc[(N, t), "CI_upper_bound"] = CI_collection[N][t][1].item()

# For filenaming so as not to overwrite previous output 
month = datetime.datetime.now().month
date = datetime.datetime.now().day
hour = datetime.datetime.now().hour
minute = datetime.datetime.now().minute
date_string = f"{month}-{date}_{hour}h{minute}m"
# Save the DataFrame
file_name = f"EPO_df_with_CI_{data_name}_{scheme}_{comparison_method}_{date_string}"
file_location = "../output"
epo_df.to_csv(f"{file_location}/{file_name}.csv")