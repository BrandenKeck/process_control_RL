# Import external libraries
import numpy as np
from keras_neural_network import neural_network

# Custom Actor / Critic with a Softmax Distribution Policy
class normal_policy_actor_critic():
    
    def __init__(self, lr, df, eql, sl,
                 critic_training_epochs = 10, critic_steps_per_epoch = None,
                 starting_variance=20, ending_variance=1e-12, ornstein_uhlenbeck_variance=1):

        # General Learning Settings
        self.lr_mu = lr
        self.df = df
        self.episode_queue_length = eql
        self.episode_queue = [episode()]
        self.last_state_terminal = True

        # Initialize Policy Objects
        self.mu = 0
        self.var = starting_variance
        self.actor_params = np.zeros(sl)
        self.variance_annealing_factor = 0.4
        self.variance_annealing_min = ending_variance
        self.variance_annealing_max = starting_variance
        self.ou_process = ornstein_uhlenbeck_process(0, ornstein_uhlenbeck_variance)
        
        # Initialize Neural Networks
        self.critic_network = neural_network(layersizes = [sl, 128, 64, 1],
                                             activations = ['relu', 'relu', 'linear'],
                                             learning_rate = 1e-3,
                                             training_epochs = critic_training_epochs,
                                             steps_per_epoch = critic_steps_per_epoch)

    def act(self, state, ornstein_uhlenbeck, learn):
        if ornstein_uhlenbeck:
            N = self.ou_process.simulate()
            self.mu = 100*clipped_sigmoid(np.dot(self.actor_params, state)) + N
            if learn:
                next_action = np.random.normal(self.mu, self.var)
            else:
                next_action = np.random.normal(self.mu, self.variance_annealing_min)
        else:
            self.mu = 100*clipped_sigmoid(np.dot(self.actor_params, state))
            if learn:
                next_action = np.random.normal(self.mu, self.var)
            else:
                next_action = np.random.normal(self.mu, self.variance_annealing_min)
        
        return next_action

    def learn(self, next_state, next_state_terminal, next_reward, last_action):

        # Apply Gradient Algorithm if start of a new episode
        if self.last_state_terminal and len(self.episode_queue) > 1:
            
            self.last_state_terminal = False
            for i in np.arange(len(self.episode_queue)-1):
                
                critic_labels = []
                e = self.episode_queue[i]
                for t in np.arange(1, len(e.rewards) - 1):

                    # Setup calculations
                    state = np.array(e.states[t])
                    state_value = self.critic_network.predict_network(np.array(state).reshape((1, -1)), np.array([0]).reshape((1, 1))).reshape((-1,))
                    next_state = np.array(e.states[t+1])
                    next_state_value = self.critic_network.predict_network(np.array(next_state).reshape((1, -1)), np.array([0]).reshape((1, 1))).reshape((-1,))
                    rewards = e.rewards[t+1]
                    
                    # Calculate and Update actor gradient
                    action = e.actions[t]
                    delta = rewards + self.df * next_state_value - state_value
                    sig = clipped_sigmoid(np.dot(self.actor_params, state))
                    d_lnpi_mu = 100*(action-100*sig/self.var**2)*(sig)*(1-sig)*state
                    self.actor_params = self.actor_params + self.lr_mu * (self.df ** t) * delta * d_lnpi_mu
                    
                    # Store Neural Network Training Data
                    critic_labels.append(rewards + self.df * next_state_value)
                    
                # Backpropagate neural networks wi
                T = len(e.rewards) - 2
                X = np.array(e.states[1:(len(e.rewards)-1)]).reshape((T,-1))
                Y = np.array(critic_labels).reshape((T,-1))
                self.critic_network.train_network(X, Y)
            
            self.anneal_variance()

        # Append episode queues with information from the current State / Action / Reward
        self.episode_queue[len(self.episode_queue) - 1].states.append(next_state)
        self.episode_queue[len(self.episode_queue) - 1].rewards.append(next_reward)
        self.episode_queue[len(self.episode_queue) - 1].actions.append(last_action)

        # Create a new episode if the current episode has just ended
        if next_state_terminal:
            self.last_state_terminal = True
            self.ou_process.value = 0
            self.episode_queue[len(self.episode_queue) - 1].total_rewards = sum(self.episode_queue[len(self.episode_queue) - 1].rewards)
            while len(self.episode_queue) > self.episode_queue_length:
                total_rewards = []
                for i in np.arange(len(self.episode_queue) - 2):
                    total_rewards.append(self.episode_queue[i].total_rewards)
                self.episode_queue.pop(total_rewards.index(min(total_rewards)))
            self.episode_queue.append(episode())
    
    def anneal_variance(self):
        d_rwd = self.episode_queue[len(self.episode_queue) - 2].total_rewards - self.episode_queue[len(self.episode_queue) - 3].total_rewards
        if d_rwd >= -self.var:
            if self.var > self.variance_annealing_min: self.var = self.variance_annealing_factor*self.var
        else:
            if self.var < self.variance_annealing_max: self.var = self.var/self.variance_annealing_factor

# Define an Ornstein-Uhlenbeck process for exploration
class ornstein_uhlenbeck_process():
    
    def __init__(self, theta, sigma):
        self.theta = theta
        self.sigma = sigma
        self.value = 0
        
    def simulate(self, dt=1):
        self.value = self.value + self.theta*self.value*dt + self.sigma*np.sqrt(dt)*np.random.normal()
        return self.value

# Episode Class for Organization of Data
class episode():

    def __init__(self):
        self.states = []
        self.rewards = []
        self.actions = []
        self.total_rewards = 0

def clipped_sigmoid(x):
    x = np.clip(x, -100, 100)
    sig = 1/(1 + np.exp(-x))
    sig = np.minimum(sig, 1 - 1e-16)
    sig = np.maximum(sig, 1e-16)
    return sig