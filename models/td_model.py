import numpy as np

class TdModel: 
    """
    Implementation of a TD-Learning model for a fear conditioning
    task. 

    Args: 
        alpha: setting for alpha learning rate
        num_trials: number of trials to be expected in the session
        num_stimuli: number of stimuli considered. 

    Attributes:
        values: np array of num_trials + 1 x num_stimuli, value of each stimuli 
            at each trial. 
    """
    def __init__(self, alpha, num_trials, num_stimuli=3):
        self.alpha = alpha
        self.values = np.zeros((num_trials + 1, num_stimuli))
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
        self.trial_idx +=1


def calc_values_for_td_model(alpha, stimuli_idxs, rewards, num_stimuli=3):
    """
    Helper function to create a TD Model with updated values-per-trial, from a series of 
    stimuli and rewards
    """
    td_model = TdModel(alpha, len(stimuli_idxs), num_stimuli)
    for stimulus_idx, reward in zip(stimuli_idxs, rewards):
        td_model.update(stimulus_idx, reward)
    return td_model
