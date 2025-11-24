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
scheme = "quantile"
comparison_method = "all"
N_DISC_VALUES = [24, 25, 26, 27, 28, 29, 30]


## DGP 
data_name = "vahid-linear"
np.random.seed(42)
n, d = 2000, 3
X = np.random.normal(1, 1, size=(n, d)).astype(np.float32)
T = (X[:, 0] - X[:, 1] + 2 * X[:, 2] + 2 + np.random.normal(0, 3, size=n)).astype(np.float32)
T = T - T.min() # Rescale
T = T / T.max() # Rescale
Y = (3 * X[:, 0] + X[:, 1] - 0.5 * X[:, 2] + 3 * T + np.random.normal(0, 2, size=n)).astype(np.float32)
def drf(t): return 3.5 + 3 * t # true dose-response function

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
file_location = "../output/11-24"
epo_df.to_csv(f"{file_location}/{file_name}.csv")