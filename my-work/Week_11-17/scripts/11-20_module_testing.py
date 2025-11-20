import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import sys
sys.path.append("../discretized-causalpfn")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from discretize import DataDiscretizer
from inference import DiscreteCausalPFN

# Parameters
scheme = "uniform"
comparison_method = "sequential"
N_DISC_VALUES = [2, 3] # [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]

# DGP
data_name = "vahid-linear"
np.random.seed(42)
n, d = 2000, 3
X = np.random.normal(1, 1, size=(n, d)).astype(np.float32)
T = (X[:, 0] - X[:, 1] + 2 * X[:, 2] + 2 + np.random.normal(0, 3, size=n)).astype(np.float32)
T = T - T.min() # Rescale
T = T / T.max() # Rescale
Y = (3 * X[:, 0] + X[:, 1] - 0.5 * X[:, 2] + 3 * T + np.random.normal(0, 2, size=n)).astype(np.float32)
def drf(t): return 3.5 + 3 * t # true dose-response function

# Inference
epos_collection = dict() # collect all results across all N_DISC in N_DISC_VALUES
for N_DISC in N_DISC_VALUES:
    discretizer = DataDiscretizer(scheme=scheme)
    T_discrete, T_vals = discretizer.discretize_treatment(T, N_DISC)
    model = DiscreteCausalPFN(
        comparison_method=comparison_method,
        N_DISC=N_DISC,
        device=device,
        verbose=True
    )
    epos_dict = model.predict_epos(
        X,
        T_discrete, 
        Y, 
        discrete_treatment_vals=T_vals,
        take_mean=True
    )
    epos_collection[N_DISC] = epos_dict

# Export data