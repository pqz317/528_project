import numpy as np
import scipy.optimize
from models.rw_model import calc_values_for_rw_model
from models.td_model import calc_values_for_td_model


def calc_rw_value_mse(alpha, stimuli_idxs, rewards, behavior):
    """
    Calculates the mean squared error of rw model value to behavioral response
    """
    model = calc_values_for_rw_model(alpha=alpha, stimuli_idxs=stimuli_idxs, rewards=rewards)
    # grab values for each of the stimuli, omit last one since it goes to num_trials + 1
    trial_vals = model.values[:-1, :]
    vals_for_stim = trial_vals[np.arange(len(trial_vals)), stimuli_idxs]
    mse = np.sum((behavior - vals_for_stim) **2) / len(behavior)
    return mse

def calc_rw_rpe_mse(alpha, stimuli_idxs, rewards, behavior):
    """
    Calculates the mean squared error of rw model rpe to behavioral response
    """
    model = calc_values_for_rw_model(alpha=alpha, stimuli_idxs=stimuli_idxs, rewards=rewards)
    # grab values for each of the stimuli, omit last one since it goes to num_trials + 1
    trial_rpes = model.rpes[:-1, :]
    rpes_for_stim = trial_rpes[np.arange(len(trial_rpes)), stimuli_idxs]
    mse = np.sum((behavior - rpes_for_stim) **2) / len(behavior)
    return mse

def calc_td_rpe_mse(alpha, k, stimuli_idxs, rewards, behavior):
    """
    Calculates the mean squared error of td model rpe to behavioral response
    """
    model = calc_values_for_td_model(alpha=alpha, k=k, gamma=1, stimuli_idxs=stimuli_idxs, rewards=rewards)
    # grab values for each of the stimuli, omit last one since it goes to num_trials + 1
    trial_rpes = model.rpes[:-1, :, :]
    rpes_for_stim = trial_rpes[np.arange(len(trial_rpes)), :, stimuli_idxs]
    mse = np.sum((behavior - rpes_for_stim) **2) / len(behavior)
    return mse

def calc_td_value_mse(alpha, k, stimuli_idxs, rewards, behavior):
    """
    Calculates the mean squared error of td model rpe to behavioral response
    """
    model = calc_values_for_td_model(alpha=alpha, k=k, gamma=1, stimuli_idxs=stimuli_idxs, rewards=rewards)
    # grab values for each of the stimuli, omit last one since it goes to num_trials + 1
    trial_vals = model.values[:-1, :, :]
    vals_for_stim = trial_vals[np.arange(len(trial_vals)), :, stimuli_idxs]
    mse = np.sum((behavior - vals_for_stim) **2) / len(behavior)
    return mse

def fit_rw_model(func, stimuli_idxs, rewards, behavior, initial_guess):
    """
    Fits a rw-learning model to behavioral response by adjusting
    free parameter alpha (learning rate) to minimize the mean 
    squared error (MSE) of rw model value to behavioral response

    Args:
        stumli_idxs: np array of floats, the index of the stimulus 
            presented at each trial
        rewards: np array of floats, the amount of rewards at each trial
        behavior: np array of floats, the behavioral response
    Returns: 
        OptimizeResult object with minimized fun value, alpha parameter. 
    """
    opt_res = scipy.optimize.minimize(
        fun=lambda x: func(x, stimuli_idxs, rewards, behavior), 
        x0=initial_guess,
        # bounds matching description in Bishara et al
        bounds=[(0, 1)],
        method="Nelder-Mead",
    )
    return opt_res


def fit_td_model(func, stimuli_idxs, rewards, behavior, initial_guess):
    min_func = None
    opt_res = None
    opt_k = None
    for k in np.arange(1, 30):
        res = scipy.optimize.minimize(
            fun=lambda x: func(x, k, stimuli_idxs, rewards, behavior), 
            x0=initial_guess,
            # bounds matching description in Bishara et al
            bounds=[(0, 1)],
            method="Nelder-Mead",
        )
        if min_func is None or res.fun < min_func:
            min_func = res.fun
            opt_res = res
            opt_k = k
    return opt_res, opt_k