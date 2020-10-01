# Import external libraries
import numpy as np
import tensorflow as tf

# Custom REINFORCE With a Binomial Distribution Policy
class Binomial_Policy_REINFORCE():

    def __init__(self, lr, df, eql, sl, epsilon=0.01):

        # General Learning Settings
        self.learning_rate = lr
        self.discount_factor = df
        self.episode_queue_length = eql
        self.episode_queue = [episode()]
        self.last_state_terminal = True

        # Initialize Policy Objects
        self.e = epsilon
        self.params = np.random.rand(sl)

    def act(self, state):
        p = sigmoid(np.dot(self.params, state))
        return (self.epsilon * (np.random.binomial(2, p) - 1))

    def learn(self, next_state, next_state_terminal, next_reward, last_action):

        # Apply Gradient Algorithm if start of a new episode
        if self.last_state_terminal and len(self.episode_queue) > 1:
            self.last_state_terminal = False
            for e in self.episode_queue:
                for t in np.arange(len(e.rewards) - 1):

                    # Calculate returns
                    returns = 0
                    for tt in np.flip(np.arange(len(e.rewards))):
                        if tt > t: returns = returns + (self.discount_factor ** (tt - t - 1)) * e.rewards[tt]
                        else: break

                    # Calculate Gradient of the Policy w.r.t. its parameters
                    state = e.states[t]
                    action = e.actions[t + 1]
                    a = np.dot(self.a_params, state)
                    b = np.dot(self.b_params, state)
                    d_lnpi_a = np.array(state)/(b - a)**3
                    d_lnpi_b = -np.array(state)/(b - a)**3

                    # Update training weights
                    self.a_params = self.a_params + self.learning_rate * (self.discount_factor ** t) * returns * d_lnpi_a
                    self.b_params = self.b_params + self.learning_rate * (self.discount_factor ** t) * returns * d_lnpi_b

            print(a)
            print(b)

        # Append episode queues with information from the current State / Action / Reward
        self.episode_queue[len(self.episode_queue) - 1].states.append(next_state)
        self.episode_queue[len(self.episode_queue) - 1].rewards.append(next_reward)
        self.episode_queue[len(self.episode_queue) - 1].actions.append(last_action)

        # Create a new episode if the current episode has just ended
        if next_state_terminal:
            self.episode_queue.append(episode())
            self.last_state_terminal = True
            while len(self.episode_queue) > self.episode_queue_length: self.episode_queue.pop(0)

# Custom REINFORCE With a Uniform Distribution Policy
class Uniform_Policy_REINFORCE():

    def __init__(self, lr, df, eql, sl):

        # General Learning Settings
        self.learning_rate = lr
        self.discount_factor = df
        self.episode_queue_length = eql
        self.episode_queue = [episode()]
        self.last_state_terminal = True

        # Initialize Policy Objects
        self.a_params = np.random.rand(sl)
        self.b_params = np.random.rand(sl)

    def act(self, state):
        a = np.dot(self.a_params, state)
        b = np.dot(self.b_params, state)
        return np.random.uniform(a, b)

    def learn(self, next_state, next_state_terminal, next_reward, last_action):

        # Apply Gradient Algorithm if start of a new episode
        if self.last_state_terminal and len(self.episode_queue) > 1:
            self.last_state_terminal = False
            for e in self.episode_queue:
                for t in np.arange(len(e.rewards) - 1):

                    # Calculate returns
                    returns = 0
                    for tt in np.flip(np.arange(len(e.rewards))):
                        if tt > t: returns = returns + (self.discount_factor ** (tt - t - 1)) * e.rewards[tt]
                        else: break

                    # Calculate Gradient of the Policy w.r.t. its parameters
                    state = e.states[t]
                    action = e.actions[t + 1]
                    a = np.dot(self.a_params, state)
                    b = np.dot(self.b_params, state)
                    d_lnpi_a = np.array(state)/(b - a)**3
                    d_lnpi_b = -np.array(state)/(b - a)**3

                    # Update training weights
                    self.a_params = self.a_params + self.learning_rate * (self.discount_factor ** t) * returns * d_lnpi_a
                    self.b_params = self.b_params + self.learning_rate * (self.discount_factor ** t) * returns * d_lnpi_b

            print(a)
            print(b)

        # Append episode queues with information from the current State / Action / Reward
        self.episode_queue[len(self.episode_queue) - 1].states.append(next_state)
        self.episode_queue[len(self.episode_queue) - 1].rewards.append(next_reward)
        self.episode_queue[len(self.episode_queue) - 1].actions.append(last_action)

        # Create a new episode if the current episode has just ended
        if next_state_terminal:
            self.episode_queue.append(episode())
            self.last_state_terminal = True
            while len(self.episode_queue) > self.episode_queue_length: self.episode_queue.pop(0)

# Custom REINFORCE With a Normal Distribution Policy
class Normal_Policy_REINFORCE():

    def __init__(self, lr, df, eql, sl):

        # General Learning Settings
        self.learning_rate_mean = lr
        self.learning_rate_var = 0
        self.discount_factor = df
        self.episode_queue_length = eql
        self.episode_queue = [episode()]
        self.last_state_terminal = True

        # Initialize Policy Objects
        self.normal_mean_params = np.zeros(sl)
        self.normal_var_params = np.zeros(sl)

    def act(self, state):
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


# Custom REINFORCE With a Normal Distribution Policy
class TF_Normal_Policy_REINFORCE():

    def __init__(self, lr, df, eql, sl):

        # General Learning Settings
        self.learning_rate_mean = lr
        self.learning_rate_var = 0
        self.discount_factor = df
        self.episode_queue_length = eql
        self.episode_queue = [episode()]
        self.last_state_terminal = True

        # Initialize Policy Objects
        self.normal_mean_params = np.zeros(sl)
        self.normal_var_params = np.zeros(sl)

    def act(self, state):
        mean = np.dot(self.normal_mean_params, state)
        var = np.exp(np.dot(self.normal_var_params, state / (np.max(np.abs(state)) + 1)))
        return np.random.normal(mean, var)

    def learn(self, next_state, next_state_terminal, next_reward, last_action):

        # Apply Gradient Algorithm if start of a new episode
        if self.last_state_terminal and len(self.episode_queue) > 1:
            self.last_state_terminal = False
            for e in self.episode_queue:
                for t in np.arange(len(e.rewards) - 1):

                    # Calculate returns
                    returns = 0
                    for tt in np.flip(np.arange(len(e.rewards))):
                        if tt > t:
                            returns = returns + (self.discount_factor ** (tt - t - 1)) * e.rewards[tt]
                        else:
                            break

                    # Calculate Gradient of the Policy w.r.t. its parameters
                    state = e.states[t]
                    action = e.actions[t + 1]
                    mean = np.dot(self.normal_mean_params, state)
                    var = np.exp(np.dot(self.normal_var_params, state / (np.max(np.abs(state)) + 1)))
                    d_lnpi_mean = (1 / var ** 2) * (action - mean) * np.array(state)
                    d_lnpi_var = (((action - mean) ** 2 / (var ** 2)) - 1) * np.array(state)

                    # Update training weights
                    self.normal_mean_params = self.normal_mean_params + self.learning_rate_mean * (
                                self.discount_factor ** t) * returns * d_lnpi_mean
                    self.normal_var_params = self.normal_var_params + self.learning_rate_var * (
                                self.discount_factor ** t) * returns * d_lnpi_var

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

def sigmoid(self, x):
    return 1/(1 + np.exp(-x))