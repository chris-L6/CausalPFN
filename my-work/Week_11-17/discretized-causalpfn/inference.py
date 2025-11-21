# Eventually want to output all predicted EPOs (maybe CEPOs) and CIs too

import numpy as np
import sys
sys.path.append("../../..")

from src.causalpfn.causal_estimator import CausalEstimator
from discretize import SCHEMES, DataDiscretizer

COMPARISON_METHODS = ["sequential", "all"]

class DiscreteCausalPFN:
    """Class for discretely predicting the dose-response curve f(t) = E[Y_t].
    Input the already-discretized treatment data T_discrete as well as the list of discrete
    treatment values T_vals. The N_DISC is fixed for any given single instance of this class.

    Args:
        device (str): The device to run the model on (e.g. 'cuda' or 'cpu)
        verbose (bool): Whether to print progress messages for the `predict_epo` function
    """
    def __init__(
            self, 
            comparison_method: str,
            N_DISC: int, 
            device: str, 
            verbose: bool
    ):
        if comparison_method not in COMPARISON_METHODS:
            raise ValueError(f"comparison_method must be one of {COMPARISON_METHODS}")
        if type(N_DISC) is not int:
            raise TypeError("N_DISC must be an int.")
        self.comparison_method = comparison_method
        self.N_DISC = N_DISC

        self.device = device
        self.verbose = verbose
        self.temperature = 1.0
        
    def predict_epos(
            self, 
            X: np.ndarray,
            T: np.ndarray,
            Y: np.ndarray,
            discrete_treatment_vals: np.ndarray,
            take_mean: bool | str = False,
            estimate_CI: bool = False
    ):
        """Predicts EPOs given data.
        
        Calls the CausalPFN _predict_cepo function and then averages across all samples to get 
        the EPO (rather than conditional EPO). Note that this is predict EPO, *not* predict CEPO.
        This function calls the CausalEstimator constructor, the fit function, and the _predict_cepo 
        function all in one, because we're not trying to train this model in detail.

        This function predicts EPOs on the full treatment range, given a FIXED N_DISC (this function
        does not loop through all N in range(N_DISC), for instance).

        Two options: "sequential" or "all". "sequential" evaluates CEPOs on sequential bins, while
        "all" evaluates CEPOs on all possible (N choose 2) pairs of bins. 

        Args: 
            X: (np.ndarray) The covariates
            T: (np.ndarray) The treatments, ALREADY DISCRETIZED
            Y: (np.ndarray) The outcomes
            discrete_treatment_vals (np.ndarray): The discrete treatment values tested
        """
        # Consistency checks
        if self.N_DISC != len(discrete_treatment_vals):
            raise ValueError(f"self.N_DISC = {self.N_DISC} but discrete_treatment_vals = {discrete_treatment_vals}; should be equal.")
        if not np.allclose(np.sort(np.unique(T)), discrete_treatment_vals, rtol=1e-10, atol=1e-5):
            raise ValueError("Treatment T does not match discrete_treatment_vals.")
        if take_mean not in [True, False, "both"]:
            raise ValueError(f"take_mean must be bool or 'both', got {take_mean}.")
        
        ## Main inference loop calling CausalPFN's CausalEstimator
        # Get the bin pairs to pass to CausalPFN via _predict_mu_0_and_mu_1
        bin_pairs = []
        if self.comparison_method == "sequential": 
            bin_pairs = list(zip(range(self.N_DISC - 1), range(1, self.N_DISC)))
        elif self.comparison_method == "all":
            for i in range(self.N_DISC):
                for j in range(i + 1, self.N_DISC):
                    bin_pairs.append([i, j])
        # For each bin pair, record the discretized treatment value, predict mu_0 and mu_1, and record these values
        epos_dict = {} # {t_val: epos}, where epos = list of all epo estimates for that t_val
        for bin_pair in bin_pairs:
            bin_0, bin_1 = bin_pair
            t_0 = discrete_treatment_vals[bin_0]
            t_1 = discrete_treatment_vals[bin_1]
            # filter X, T, Y values to those included in bin_pair
            ids = (np.abs(T - t_0) < 1e-4) | (np.abs(T - t_1) < 1e-4)
            T_temp = np.where(np.abs(T[ids] - t_0) < 1e-4, 0, 1).astype(np.float32)
            X_temp = X[ids].astype(np.float32)
            Y_temp = Y[ids].astype(np.float32)
            mu_0, mu_1 = self._predict_mu_0_and_mu_1(X_temp, T_temp, Y_temp)
            # record epos for t_0, t_1
            if t_0 not in epos_dict: 
                epos_dict[t_0] = [mu_0]
            else:
                epos_dict[t_0].append(mu_0)
            if t_1 not in epos_dict: 
                epos_dict[t_1] = [mu_1]
            else:
                epos_dict[t_1].append(mu_1)

        if take_mean == "both":
            epos_means = dict()
            for t_val in epos_dict:
                epos_means[t_val] = np.mean(epos_dict[t_val])
            return epos_dict, epos_means
        elif type(take_mean) is not bool:
            raise ValueError(f"Invalid entry for take_mean: {take_mean}")
        else:
            if not take_mean:
                return epos_dict
            else:
                epos_means = dict()
                for t_val in epos_dict:
                    epos_means[t_val] = np.mean(epos_dict[t_val])
                return epos_means
        
    def _predict_mu_0_and_mu_1(
            self, 
            X: np.ndarray, 
            T: np.ndarray, 
            Y: np.ndarray 
    ):
        """Given *binary* treatment data T (pre-processed to be 0s and 1s), predicts the EPOs
        mu_0 and mu_1.

        Args:
            X: (np.ndarray) The covariates
            T: (np.ndarray) The treatments, binarized to be in {0, 1}
            Y: (np.ndarray) The outcomes
        """
        # Check that T is binarized and has only 0s and 1s
        T_vals = np.sort(np.unique(T))
        if not np.array_equal(T_vals, np.array([0, 1])):
            raise ValueError(f"T must be binarized with values 0 and 1, but is {T_vals}") 

        X_context, t_context, y_context = X, T, Y
        X_query = X
        t_all_ones = np.ones(X_query.shape[0], dtype=X_query.dtype)
        t_all_zeros = np.zeros(X_query.shape[0], dtype=X_query.dtype)
        model = CausalEstimator(
            device=self.device,
            verbose=self.verbose
        )
        model.fit(X, T, Y)
        mu_vals = model._predict_cepo(
            X_context=X_context, 
            t_context=t_context, 
            y_context=y_context,
            X_query=np.concatenate([X_query, X_query], axis=0),
            t_query=np.concatenate([t_all_zeros, t_all_ones], axis=0),
            temperature=self.temperature
        )
        mu_0 = (mu_vals[: X_query.shape[0]]).mean()
        mu_1 = (mu_vals[X_query.shape[0] :]).mean()
        return mu_0, mu_1