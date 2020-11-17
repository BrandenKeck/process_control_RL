# Import external libraries
import numpy as np
from keras_neural_network import neural_network

# Custom Actor / Critic with a Softmax Distribution Policy
class normal_policy_actor_critic():
    
    def __init__(self, lr, df, eql, sl, starting_variance=20, ending_variance=1e-12, ou_variance=1):

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
        self.ou_process = ornstein_uhlenbeck_process(0, ou_variance)
        
        # Initialize Neural Networks
        self.critic_network = critic_neural_network([sl, 128, 64, 32, 1], 1e-5, False)

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
            for e in self.episode_queue:
                for t in np.arange(1, len(e.rewards) - 1):

                    # Setup calculations
                    state = np.array(e.states[t])
                    state_value = self.critic_network.evaluate(np.array(state))
                    next_state = np.array(e.states[t+1])
                    next_state_value = self.critic_network.evaluate(np.array(next_state))
                    rewards = e.rewards[t+1]
                    
                    # Calculate gradient
                    action = e.actions[t]
                    sig = clipped_sigmoid(np.dot(self.actor_params, state))
                    mu = 100*sig
                    d_lnpi_mu = 100*(action-mu/self.var**2)*(sig)*(1-sig)*state

                    # Update training weights
                    Y = rewards + self.df * next_state_value
                    Y_hat = state_value
                    delta = Y - Y_hat
                    #print("V: " + str(state_value))
                    #print("R: " + str(rewards))
                    print("D: " + str(delta))
                    #print("")
                    self.actor_params = self.actor_params + self.lr_mu * (self.df ** t) * delta * d_lnpi_mu
                    
                    # Backpropagate neural networks
                    self.critic_network.backpropagate(np.array(state), Y, Y_hat)
            
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


# Custom Net for the Policy Gradient Learning Method
# All hidden layers are ReLU.  Output is either linear or sigmoid.
# This is controlled by self.sigmoid_output
class critic_neural_network():
    
    def __init__(self, layersizes, learning_rate=0.01, sigmoid=True):
        
        # Configurable Settings
        self.layersizes = np.array([int(i) for i in layersizes])
        self.learning_rates = learning_rate * np.ones(len(self.layersizes)-1)
        self.huber_cost_delta = 0.01

        # Initialize Node Value and Weight Arrays
        self.z = []
        self.a = []
        self.w = []
        self.b = []
        for i in np.arange(1, len(self.layersizes)):
            self.w.append(0.01 * np.random.randn(self.layersizes[i], self.layersizes[i-1]))
            self.b.append(np.zeros(self.layersizes[i]))
            
    # Forward calculation of the output
    def evaluate(self, X):
        
        # Set Input Data to Initially Activated Layer
        self.a = [X.reshape(-1, 1)]
        self.z = []

        # Loop over all layers
        for i in np.arange(len(self.layersizes) - 2):

            # Calculate Z (pre-activated node values for hidden layers)
            zz = np.matmul(self.w[i], self.a[i]) + np.broadcast_to(self.b[i], (1, self.b[i].shape[0])).transpose()
            self.z.append(zz)

            # Calculate A (ReLU activated node values for hidden layers)
            self.a.append(self.ReLU(zz))
        
        # Compute the output layer pre-activated node and activated node values
        zz = np.matmul(self.w[len(self.w) - 1], self.a[len(self.a) - 1])
        self.z.append(zz)
        self.a.append(self.linear(zz))
        return self.a[len(self.a) - 1][0][0]
        
    # Backpropagation / Training Function
    # Cost is calculated per the policy gradient method and passed to this function as an input
    def backpropagate(self, X, Y, Y_hat):
        
        # Evaluate the input to compute activated node values
        self.evaluate(X)
        
        # Evaluate output change and combine with loss derivative
        dL = self.d_huber(Y, Y_hat, self.huber_cost_delta)
        dA = self.d_linear(self.z[len(self.z)-1])
        #print("dL/dA: " + str(dL))
        dz = dL * dA
        prev_dz = dz
        
        # Train the outermost layer of the network
        dw = np.matmul(dz, self.a[len(self.a)-1].T)
        db = (np.sum(dz, axis=1, keepdims=True)).reshape((self.b[len(self.b)-1].shape[0],))
        self.w[len(self.w)-1] = self.w[len(self.w)-1] - self.learning_rates[len(self.w)-1] * dw
        self.b[len(self.b)-1] = self.b[len(self.b)-1] - self.learning_rates[len(self.b)-1] * db

        # Loop over layers backwards
        for i in np.flip(np.arange(len(self.w)-1)):
            
            dA = self.d_ReLU(self.z[i])
            dz = np.matmul(self.w[i + 1].T, prev_dz) * dA
            prev_dz = dz

            # Calculate Weight Derivatives
            dw = np.matmul(dz, self.a[i].T)
            db = (np.sum(dz, axis=1, keepdims=True)).reshape((self.b[i].shape[0],))
            self.w[i] = self.w[i] - self.learning_rates[i] * dw
            self.b[i] = self.b[i] - self.learning_rates[i] * db
    
    def ReLU(self, x): return np.maximum(0, x)
    def linear(self, x): return x
    def d_ReLU(self, x): return np.where(x > 0, 1.0, 0)
    def d_linear(self, x): return np.ones(x.shape)
    def d_huber(self, Y, Y_hat, delta=1):
        if np.linalg.norm(Y_hat - Y) < delta: return Y_hat - Y
        else: return delta * np.sign(Y_hat - Y)

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