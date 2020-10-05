# Import external libraries
import numpy as np

# Custom Actor / Critic with a Binomial Distribution Policy
class Binomial_Policy_Actor_Critic():

    def __init__(self, lr, df, eql, sl):

        # General Learning Settings
        self.lr_p = lr
        self.lr_vf = lr
        self.df = df
        self.episode_queue_length = eql
        self.episode_queue = [episode()]
        self.last_state_terminal = True

        # Initialize Policy Objects
        self.p = 0
        self.e = 0
        self.params = np.zeros(sl)
        self.w = np.zeros(sl)

    def act(self, state, last_action):
        self.p = sigmoid(np.dot(self.params, state))
        print(self.p)
        self.e = 0.1
        next_action = last_action + (self.e * (np.random.binomial(2, self.p) - 1))
        return next_action

    def learn(self, next_state, next_state_terminal, next_reward, last_action):

        # Apply Gradient Algorithm if start of a new episode
        if self.last_state_terminal and len(self.episode_queue) > 1:
            self.last_state_terminal = False
            for e in self.episode_queue:
                for t in np.arange(1, len(e.rewards) - 1):

                    # Setup calculations
                    state = np.array(e.states[t])
                    state_value = np.dot(state, self.w)
                    next_state = np.array(e.states[t+1])
                    next_state_value = np.dot(next_state, self.w)
                    rewards = e.rewards[t+1]

                    # Calculate gradient
                    d_action = e.actions[t] - e.actions[t - 1]
                    sig = sigmoid(np.dot(self.params, state))
                    if d_action < 0:
                        d_lnpi = -(2 * sig) * np.array(state)
                    elif d_action == 0:
                        d_lnpi = (1 - 2 * sig) * np.array(state)
                    elif d_action > 0:
                        d_lnpi = 2 * (1 - sig) * np.array(state)

                    # Update training weights
                    delta = rewards + self.df * next_state_value - state_value
                    print(state_value)
                    self.w = self.w + self.lr_vf * delta * state
                    self.params = self.params + self.lr_p * (self.df ** t) * delta * d_lnpi

        # Append episode queues with information from the current State / Action / Reward
        self.episode_queue[len(self.episode_queue) - 1].states.append(next_state)
        self.episode_queue[len(self.episode_queue) - 1].rewards.append(next_reward)
        self.episode_queue[len(self.episode_queue) - 1].actions.append(last_action)

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

def sigmoid(x):
    x = np.clip(x, -100, 100)
    sig = 1/(1 + np.exp(-x))
    sig = np.minimum(sig, 1 - 1e-16)
    sig = np.maximum(sig, 1e-16)
    return sig