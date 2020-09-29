# Import external libraries
import numpy as np
from copy import deepcopy

# Primary Class used to construct PG Methods
class REINFORCE():

    def __init__(self, lr_mean, lr_var, df, eql, sl):

        # General Learning Settings
        self.learning_rate_mean = lr_mean
        self.learning_rate_var = lr_var
        self.discount_factor = df
        self.episode_queue_length = eql
        self.episode_queue = [episode()]
        self.last_state_terminal = True

        # Set Policy Type
        self.use_normal_policy = True

        # Initialize Policy Objects
        self.normal_mean_params = np.zeros(sl)
        self.normal_var_params = np.zeros(sl)

    def act(self, state):
        if self.use_normal_policy:
            mean = np.dot(self.normal_mean_params, state)
            var = np.exp(np.dot(self.normal_var_params, state/(np.max(np.abs(state))+1)))
            return np.random.normal(mean, var)
        
    def learn(self, next_state, next_state_terminal, next_reward, last_action):
        
        # Apply Gradient Algorithm if start of a new episode
        if self.last_state_terminal and len(self.episode_queue) > 1:
            self.last_state_terminal = False
            for e in self.episode_queue:
                for t in np.arange(len(e.rewards)-1):
                    
                    # Calculate returns
                    returns = 0
                    for tt in np.flip(np.arange(len(e.rewards))):
                        if tt > t: returns = returns + (self.discount_factor**(tt - t - 1)) * e.rewards[tt]
                        else: break

                    # Calculate Gradient of the Policy w.r.t. its parameters
                    state = e.states[t]
                    action = e.actions[t+1]
                    mean = np.dot(self.normal_mean_params, state)
                    var = np.exp(np.dot(self.normal_var_params, state/(np.max(np.abs(state))+1)))
                    d_lnpi_mean = (1 / var**2) * (action - mean) * np.array(state)
                    d_lnpi_var = (((action - mean)**2/(var**2)) - 1) * np.array(state)

                    # Update training weights
                    self.normal_mean_params = self.normal_mean_params + self.learning_rate_mean*(self.discount_factor**t)*returns*d_lnpi_mean
                    self.normal_var_params = self.normal_var_params + self.learning_rate_var*(self.discount_factor**t)*returns*d_lnpi_var
        
        # Append episode queues with information from the current State / Action / Reward
        self.episode_queue[len(self.episode_queue)-1].states.append(next_state)
        self.episode_queue[len(self.episode_queue)-1].rewards.append(next_reward)
        self.episode_queue[len(self.episode_queue)-1].actions.append(last_action)

        # Create a new episode if the current episode has just ended
        if next_state_terminal: 
            self.episode_queue.append(episode())
            self.last_state_terminal = True
            while len(self.episode_queue) > self.episode_queue_length: self.episode_queue.pop(0)

        

# Episode Class for Organization of Data
class episode():

    def __init__(self):
        self.states = []
        self.rewards = []
        self.actions = []