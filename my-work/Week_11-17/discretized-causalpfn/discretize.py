import numpy as np

# Implemented discretization schemes
SCHEMES = ["uniform", "quantile"]

class DataDiscretizer:
    """Class for discretizing treatment variable based on passed scheme.
    """
    def __init__(self, scheme: str):
        if scheme not in SCHEMES:
            raise ValueError(f"scheme must be one of {SCHEMES}")
        self.scheme = scheme
    
    def discretize_treatment(self, T: np.ndarray, N: int) -> tuple[np.ndarray, np.ndarray]:
        """Function that actually discretizes the treatment data. 

        Args:
            T (np.ndarray): The treatment data to be discretized
            N (int): The number of bins (sub-intervals of [0, 1]) to discretize into

        Returns:
            np.ndarray: The discretized treatment data
            np.ndarray: The discrete treatment values
        """
        if self.scheme == "uniform": 
            dt = 1 / (N - 1)
            T_discrete = np.round(np.array(T, dtype=np.float32) / dt) * dt
            T_vals = np.linspace(0, 1, N)

            return T_discrete, T_vals
        
        if self.scheme == "quantile":
            T_discrete = np.zeros(T.shape)
            T_vals = np.zeros((N,))
            bin_edges = np.percentile(T, np.linspace(0, 100, N + 1))
            for i in range(len(bin_edges) - 1):
                left_edge = bin_edges[i]
                right_edge = bin_edges[i + 1]
                ids = (T >= left_edge) & (T <= right_edge)
                avg = np.mean(T[ids])
                T_discrete[ids] = avg
                T_vals[i] = avg

            return T_discrete, T_vals
        
        else: 
            raise ValueError(f"self.scheme must be one of {SCHEMES}")