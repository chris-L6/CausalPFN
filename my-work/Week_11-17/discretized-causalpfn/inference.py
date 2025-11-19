# Feed T_discrete to the object; the object should determine the N_DISC used (by looking at len(T_vals))
# Eventually want to output all predicted EPOs (maybe CEPOs) and CIs too

from multiprocessing import Value
import numpy as np
import sys
sys.path.append("../../..")

from src.causalpfn.causal_estimator import CausalEstimator
from discretize import SCHEMES, DataDiscretizer

class DiscreteCausalPFN:
    """Class for discretely predicting the dose-response curve f(t) = E[Y_t].
    Input the already-discretized treatment data T_discrete as well as the list of discrete
    treatment values T_vals. The N_DISC is fixed for any given single instance of this class.

    Args:
        device (str): The device to run the model on (e.g. 'cuda' or 'cpu)
        verbose (bool): Whether to print progress messages for the `predict_epo` function
    """
    def __init__(self, scheme: str, N_DISC: int, device: str, verbose: bool):
        if scheme not in SCHEMES:
            raise ValueError(f"scheme must be one of {SCHEMES}")
        if type(N_DISC) is not int:
            raise TypeError("N_DISC must be an int.")
        self.scheme = scheme
        self.N_DISC = N_DISC

        self.device = device
        self.verbose = verbose
        self.temperature = 1.0
        
    def predict_epo(
            self, 
            X: np.ndarray,
            T: np.ndarray,
            Y: np.ndarray,
            discrete_treatment_vals: np.ndarray
    ):
        """Predicts EPOs given data.
        
        Calls the CausalPFN _predict_cepo function and then averages across all samples to get 
        the EPO (rather than conditional EPO). Note that this is predict EPO, *not* predict CEPO.
        This function calls the CausalEstimator constructor, the fit function, and the _predict_cepo 
        function all in one, because we're not trying to train this model in detail.

        Args: 
            X: (np.ndarray) The covariates
            T: (np.ndarray) The treatments, ALREADY DISCRETIZED
            Y: (np.ndarray) The outcomes
            discrete_treatment_vals (np.ndarray): The discrete treatment values tested
        """
        # Consistency checks
        if self.N_DISC != len(discrete_treatment_vals):
            raise ValueError(f"self.N_DISC = {self.N_DISC} but discrete_treatment_vals = {discrete_treatment_vals}; should be equal.")
        if np.sort(np.unique(T)) != discrete_treatment_vals:
            raise ValueError("Treatment T does not match discrete_treatment_vals.")
        
        # Main inference loop calling CausalPFN's CausalEstimator
        list_of_epos = []
        