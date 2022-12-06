import numpy as np

class RWModel: 
    """
    Implementation of a Rescorla Wagner-Learning model for a fear conditioning
    task. 

    Args: 
        alpha: setting for alpha learning rate
        num_trials: number of trials to be expected in the session
        num_stimuli: number of stimuli considered. 
        init_values: np array of num_stimuli length, which values to start, 0 if None
s
    Attributes:
        values: np array of num_trials + 1 x num_stimuli, value of each stimuli 
            at each trial. 
    """
    def __init__(self, alpha, num_trials, num_stimuli=3, init_values=None):
        self.alpha = alpha
        self.values = np.zeros((num_trials + 1, num_stimuli))
        if init_values:
            self.values[0, :] = init_values
        self.rpes = np.zeros((num_trials + 1, num_stimuli))
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
        self.values[self.trial_idx + 1, :] = self.values[self.trial_idx, :]

        # delta or TD-error or RPE is difference between reward and current value
        # of stimulus
        delta = reward - self.values[self.trial_idx, stimulus_idx]

        # update value of stimulus with delta x learning rate
        self.values[self.trial_idx + 1, stimulus_idx] += self.alpha * delta
        self.rpes[self.trial_idx, stimulus_idx] = delta
        self.trial_idx +=1


def calc_values_for_rw_model(alpha, stimuli_idxs, rewards, num_stimuli=3, init_vals=None):
    """
    Helper function to create a TD Model with updated values-per-trial, from a series of 
    stimuli and rewards
    """
    td_model = RWModel(alpha, len(stimuli_idxs), num_stimuli, init_vals)
    for stimulus_idx, reward in zip(stimuli_idxs, rewards):
        td_model.update(stimulus_idx, reward)
    return td_model

