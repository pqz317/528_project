import numpy as np
import scipy.optimize
from models.td_model import calc_values_for_td_model


def calc_td_mse(alpha, stimuli_idxs, rewards, behavior):
    """
    Calculates the mean squared error of TD model value to behavioral response
    """
    model = calc_values_for_td_model(alpha=alpha, stimuli_idxs=stimuli_idxs, rewards=rewards)
    # grab values for each of the stimuli, omit last one since it goes to num_trials + 1
    trial_vals = model.values[:-1, :]
    vals_for_stim = trial_vals[np.arange(len(trial_vals)), stimuli_idxs]
    mse = np.sum((behavior - vals_for_stim) **2) / len(behavior)
    return mse


def get_td_mse_func(stimuli_idxs, rewards, behavior):
    """
    Returns back a function to be optimized wrt to TD model learning rate
    """
    return lambda alpha: calc_td_mse(alpha, stimuli_idxs, rewards, behavior)


def fit_model(stimuli_idxs, rewards, behavior, initial_guess):
    """
    Fits a TD-learning model to behavioral response by adjusting
    free parameter alpha (learning rate) to minimize the mean 
    squared error (MSE) of TD model value to behavioral response

    Args:
        stumli_idxs: np array of floats, the index of the stimulus 
            presented at each trial
        rewards: np array of floats, the amount of rewards at each trial
        behavior: np array of floats, the behavioral response
    Returns: 
        OptimizeResult object with minimized fun value, alpha parameter. 
    """
    opt_res = scipy.optimize.minimize(
        fun=get_td_mse_func(stimuli_idxs, rewards, behavior), 
        x0=initial_guess,
        # bounds matching description in Bishara et al
        bounds=[(0, 1)],
        method="Nelder-Mead",
    )
    return opt_res