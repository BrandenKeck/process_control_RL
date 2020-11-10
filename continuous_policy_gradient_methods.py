# Import external libraries
import numpy as np
from copy import deepcopy

class ddpg():
    
    def __init__(self, lr=1e-11, df=1, trackf=1e-3, expf=0.8, eql=11, sl=10, al=1):
        
        # General Learning Settings
        self.lr = lr
        self.df = df
        self.trackf = trackf
        self.expf = expf
        self.eql = eql
        self.sl = sl
        self.al = al
        
        # Initialize Learning Objects
        self.last_state_terminal = True
        self.episode_queue = [episode()]
        self.ou_process = ornstein_uhlenbeck_process(0, expf)
        self.critic_network = policy_gradients_neural_net([sl+al, 128, 64, 1], 1e-10, False) # Critic Network
        self.actor_network = policy_gradients_neural_net([sl, 128, 64, 1], 1, True) # Actor Network
        self.critic_target_network = deepcopy(self.critic_network)
        self.actor_target_network = deepcopy(self.actor_network)
    
    def deterministic_action(self, state):
        return 100*self.actor_network.evaluate(np.array(state))
        
    def stochastic_action(self, state):
        N = self.ou_process.simulate()
        print("Last Error: " + str(state[len(state)-1]) + " | Action: " + str(self.actor_network.evaluate(np.array(state))))
        return 100*self.actor_network.evaluate(np.array(state)) + N

    def learn(self, next_state, next_state_terminal, next_reward, last_action):

        # Apply Gradient Algorithm if start of a new episode
        if self.last_state_terminal and len(self.episode_queue) > 1:
            self.last_state_terminal = False
            for e in self.episode_queue:
                for t in np.arange(1, len(e.rewards) - 1):

                    # Set Current Timestep Variables
                    action = np.array([e.actions[t]]).reshape(-1,)
                    state = np.array([e.states[t-1]]).reshape(-1,)
                    next_state = np.array([e.states[t]]).reshape(-1,)
                    rewards = e.rewards[t]
                    
                    # Evaluate Neural Networks
                    predicted_action = np.array([self.actor_network.evaluate(state)]).reshape(-1,)
                    action_value = self.critic_network.evaluate(np.concatenate((state, action)))
                    next_target_action = np.array([self.actor_target_network.evaluate(next_state)]).reshape(-1,)
                    next_target_action_value = self.critic_target_network.evaluate(np.concatenate((next_state, next_target_action)))
                    
                    # Calculate Critic Loss and Backpropagate the Actor and Critic Networks
                    delta = rewards + self.df * next_target_action_value - action_value
                    self.critic_network.backpropagate_as_critic(np.concatenate((state, action)), delta)
                    self.actor_network.backpropagate_as_actor(self.critic_network, state, predicted_action, self.sl, self.al)
                    
            # Update Target Networks at end of training steps
            for i in np.arange(len(self.actor_network.w)):
                self.actor_target_network.w[i] = self.trackf*self.actor_network.w[i] + (1 - self.trackf)*self.actor_target_network.w[i]
                self.actor_target_network.b[i] = self.trackf*self.actor_network.b[i] + (1 - self.trackf)*self.actor_target_network.b[i]
            for i in np.arange(len(self.critic_network.w)):
                self.critic_target_network.w[i] = self.trackf*self.critic_network.w[i] + (1 - self.trackf)*self.critic_target_network.w[i]
                self.critic_target_network.b[i] = self.trackf*self.critic_network.b[i] + (1 - self.trackf)*self.critic_target_network.b[i]
                
        # Append episode queues with information from the current State / Action / Reward
        self.episode_queue[len(self.episode_queue) - 1].states.append(next_state)
        self.episode_queue[len(self.episode_queue) - 1].rewards.append(next_reward)
        self.episode_queue[len(self.episode_queue) - 1].actions.append(last_action/100)

        # Create a new episode if the current episode has just ended
        if next_state_terminal:
            self.last_state_terminal = True
            self.ou_process.value = 0
            self.episode_queue[len(self.episode_queue) - 1].total_rewards = sum(self.episode_queue[len(self.episode_queue) - 1].rewards)
            self.episode_queue.append(episode())
            while len(self.episode_queue) > self.eql: self.episode_queue.pop(0)


# Custom Net for the Policy Gradient Learning Method
# All hidden layers are ReLU.  Output is either linear or sigmoid.
# This is controlled by self.sigmoid_output
class policy_gradients_neural_net():
    
    def __init__(self, layersizes, learning_rate=0.01, is_actor=True):
        
        # Configurable Settings
        self.layersizes = np.array([int(i) for i in layersizes])
        self.learning_rates = learning_rate * np.ones(len(self.layersizes)-1)
        self.is_actor = is_actor
        self.is_critic = not is_actor

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
        if self.is_actor: self.a.append(self.sigmoid(zz))
        else: self.a.append(self.linear(zz))
        return self.a[len(self.a) - 1][0][0]
        
    # Backpropagation / Training Function
    # Cost is calculated per the policy gradient method and passed to this function as an input
    def backpropagate_as_critic(self, state, dL):
        
        # Evaluate the input to compute activated node values
        self.evaluate(state)
        
        # Evaluate output change and combine with loss derivative
        dA = self.d_linear(self.z[len(self.z)-1])
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
    
    def backpropagate_as_actor(self, critic_network, state, action, sl, al):
        
        # Evaluate output change and combine with loss derivative
        self.evaluate(state)
        critic_network.evaluate(np.concatenate((state, action)))
        dz = critic_network.d_linear(critic_network.z[len(critic_network.z)-1])
        prev_dz = dz
        
        # Loop over layers backwards to get dQ/da from critic network
        for i in np.flip(np.arange(len(critic_network.w)-1)):
            dA = critic_network.d_ReLU(critic_network.z[i])
            dz = np.matmul(critic_network.w[i + 1].T, prev_dz) * dA
            prev_dz = dz
        
        # Change in the critic network with respect to the action            
        dact = dz.reshape(-1,)[sl:sl+al]
        dact = dact.reshape(dact.size, -1)
        
        # Evaluate output change and combine with loss derivative
        dA = self.d_sigmoid(self.z[len(self.z)-1])
        dz = dact * dA
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


# Define an Ornstein-Uhlenbeck process for DDPG method exploration
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