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
        
        # Initialize Neural Networks
        self.p_net = policy_gradients_neural_net([sl, 128, 64, 1], lr, True)
        self.v_net = policy_gradients_neural_net([sl, 128, 64, 1], lr, False)

    def act(self, state, last_action):
        self.p = clipped_sigmoid(np.dot(self.params, state))
        #self.p = self.p_net.evaluate(np.array(state))
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
                    #state_value = np.dot(state, self.w)
                    state_value = self.v_net.evaluate(np.array(state))
                    next_state = np.array(e.states[t+1])
                    #next_state_value = np.dot(next_state, self.w)
                    next_state_value = self.v_net.evaluate(np.array(next_state))
                    rewards = e.rewards[t+1]
                    
                    # Calculate gradient
                    d_action = e.actions[t] - e.actions[t - 1]
                    sig = clipped_sigmoid(np.dot(self.params, state))
                    if d_action < 0: d_lnpi = -(2 * sig) * np.array(state)
                    elif d_action == 0: d_lnpi = (1 - 2 * sig) * np.array(state)
                    elif d_action > 0: d_lnpi = 2 * (1 - sig) * np.array(state)

                    # Update training weights
                    delta = rewards + self.df * next_state_value - state_value
                    #self.w = self.w + self.lr_vf * delta * state
                    self.params = self.params + self.lr_p * (self.df ** t) * delta * d_lnpi
                    
                    # Backpropagate neural networks
                    #self.p_net.backpropagate(np.array(state), (self.df ** t) * delta * d_lnpi)
                    self.v_net.backpropagate(np.array(state), delta)

        # Append episode queues with information from the current State / Action / Reward
        self.episode_queue[len(self.episode_queue) - 1].states.append(next_state)
        self.episode_queue[len(self.episode_queue) - 1].rewards.append(next_reward)
        self.episode_queue[len(self.episode_queue) - 1].actions.append(last_action)

        # Create a new episode if the current episode has just ended
        if next_state_terminal:
            self.episode_queue.append(episode())
            self.last_state_terminal = True
            while len(self.episode_queue) > self.episode_queue_length: self.episode_queue.pop(0)

# Custom Net for the Policy Gradient Learning Method
# All hidden layers are ReLU.  Output is either linear or sigmoid.
# This is controlled by self.sigmoid_output
class policy_gradients_neural_net():
    
    def __init__(self, layersizes, learning_rate=0.01, sigmoid=True):
        
        # Configurable Settings
        self.layersizes = np.array([int(i) for i in layersizes])
        self.learning_rates = learning_rate * np.ones(len(self.layersizes)-1)
        self.sigmoid_output = sigmoid

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
        if self.sigmoid_output: self.a.append(self.sigmoid(zz))
        else: self.a.append(self.linear(zz))
        return self.a[len(self.a) - 1][0][0]
        
    # Backpropagation / Training Function
    # Cost is calculated per the policy gradient method and passed to this function as an input
    def backpropagate(self, X, dL):
        
        # Evaluate the input to compute activated node values
        self.evaluate(X)
        
        # Evaluate output change and combine with loss derivative
        if self.sigmoid_output: dA = self.d_sigmoid(self.z[len(self.z)-1])
        else: dA = self.d_linear(self.z[len(self.z)-1])
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
    def sigmoid(self, x): return 1/(1+np.exp(-1*x.astype(np.float32)))
    def linear(self, x): return x
    def d_ReLU(self, x): return np.where(x > 0, 1.0, 0)
    def d_linear(self, x): return np.ones(x.shape)
    def d_sigmoid(self, x): return self.sigmoid(x)*(1 - self.sigmoid(x))


# Episode Class for Organization of Data
class episode():

    def __init__(self):
        self.states = []
        self.rewards = []
        self.actions = []

def clipped_sigmoid(x):
    x = np.clip(x, -100, 100)
    sig = 1/(1 + np.exp(-x))
    sig = np.minimum(sig, 1 - 1e-16)
    sig = np.maximum(sig, 1e-16)
    return sig