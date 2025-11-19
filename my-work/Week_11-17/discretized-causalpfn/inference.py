# Feed T_discrete to the object; the object should determine the N_DISC used (by looking at len(T_vals))
# Eventually want to output all predicted EPOs (maybe CEPOs) and CIs too

class DiscreteCausalPFN:
    """Class for discretely predicting the dose-response curve f(t) = E[Y_t].
    Input the already-discretized treatment data T_discrete as well as the list of discrete
    treatment values T_vals.

    Args:
        device (str): The device to run the model on (e.g. 'cuda' or 'cpu)
        verbose (bool): Whether to print progress messages for the `predict_epo` function
    """
    def __init__(self, device: str, verbose: bool):
        self.device = device
        self.verbose = verbose

        self.X_train, self.t_train, self.y_train = None, None, None
    
    def _check_fitted(self):
        if self.X_train is None or self.t_train is None or self.y_train is None:
            raise ValueError("The estimator must be fitted before calling the estimate function.")