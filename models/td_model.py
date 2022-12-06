import numpy as np

class TDModel: 
    """
    Implementation of a TD-Learning model for a fear conditioning
    task. 

    Args: 
        alpha: setting for alpha learning rate
        k: int for how many steps forward to look 
        gamma: discount factor, how much to weigh future rewards
        num_trials: number of trials to be expected in the session
        num_time_steps: number of time steps in trial
        num_stimuli: number of stimuli considered. 
        init_values: np array of num_time_steps x num_stimuli, which values to start, 0s if None.

    Attributes:
        values: np array of num_trials + 1 x num_stimuli, value of each stimuli 
            at each trial. 
    """
    def __init__(self, alpha, k, gamma, num_trials, num_time_steps=60, num_stimuli=3, init_values=None):
        self.alpha = alpha
        self.k = k
        self.gamma = gamma
        self.values = np.zeros((num_trials + 1, num_time_steps, num_stimuli))

        if init_values:
            self.values[0, :, :] = init_values
        self.rpes = np.zeros((num_trials + 1, num_time_steps, num_stimuli))
        self.trial_idx = 0

    def update(self, stimulus_idx, reward):
        """
        Calculates and updates the values of relevant stimulus

        Args: 
            stimulus_idx: index of stimulus that was presented
            reward: amount of reward given this trial, in 
                fear conditioning case either 0 or negative number
        """
        # copy over values to the next trial, only change the value specified
        self.values[self.trial_idx + 1, :, :] = self.values[self.trial_idx, :, :]

        stim_vals = self.values[self.trial_idx, :, stimulus_idx]
        for t in range(len(stim_vals) - 1):
            t_fut = np.min((t + self.k, len(stim_vals) - 1))
            v_t = stim_vals[t]
            v_fut = stim_vals[t_fut]
            if t_fut == len(stim_vals) - 1:
                r = reward
            else:
                r = 0
            rpe = r + self.gamma * v_fut - v_t
            self.values[self.trial_idx + 1, t, stimulus_idx] = v_t + self.alpha * rpe
            self.rpes[self.trial_idx, t:t+self.k, stimulus_idx] = rpe
        self.trial_idx += 1


def calc_values_for_td_model(alpha, k, gamma, stimuli_idxs, rewards, num_time_steps=60, num_stimuli=3, init_vals=None):
    """
    Helper function to create a TD Model with updated values-per-trial, from a series of 
    stimuli and rewards
    """
    td_model = TDModel(alpha, k, gamma, len(stimuli_idxs), num_time_steps, num_stimuli, init_vals)
    for stimulus_idx, reward in zip(stimuli_idxs, rewards):
        td_model.update(stimulus_idx, reward)
    return td_model

