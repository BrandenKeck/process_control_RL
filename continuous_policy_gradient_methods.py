# Import external libraries
import numpy as np
from copy import deepcopy
from keras_neural_network import ddpg_neural_network

class ddpg():
    
    def __init__(self, actor_lr=1e-3, critic_lr=1e-3, 
                 df=1, eql=11, sl=10, al=1,
                 target_frac=1e-3, ornstein_uhlenbeck_variance=0.8,
                 actor_training_epochs = 10, actor_steps_per_epoch = None,
                 critic_training_epochs = 10, critic_steps_per_epoch = None):
        
        # General Learning Settings
        self.alr = actor_lr
        self.clr = critic_lr
        self.df = df
        self.eql = eql
        self.sl = sl
        self.al = al
        self.target_frac = target_frac
        self.ornstein_uhlenbeck_variance = ornstein_uhlenbeck_variance
       
        # Episodic Learning Components
        self.last_state_terminal = True
        self.episode_queue = [episode()]
        self.ornstein_uhlenbeck_process = ornstein_uhlenbeck_process(0, ornstein_uhlenbeck_variance)
        
        # Create Actor Neural Networks
        self.actor_network = ddpg_neural_network(layersizes = [sl, 128, 64, 1],
                                             activations = ['relu', 'relu', 'sigmoid'],
                                             learning_rate = 1e-3,
                                             training_epochs = critic_training_epochs,
                                             steps_per_epoch = critic_steps_per_epoch)
        self.actor_target_network = ddpg_neural_network(layersizes = [sl, 128, 64, 1],
                                             activations = ['relu', 'relu', 'sigmoid'],
                                             learning_rate = 1e-3,
                                             training_epochs = critic_training_epochs,
                                             steps_per_epoch = critic_steps_per_epoch)
        self.actor_target_network.model.set_weights(self.actor_network.model.get_weights())
        
        # Create Critic Neural Networks
        self.critic_network = ddpg_neural_network(layersizes = [sl+al, 128, 64, 1],
                                             activations = ['relu', 'relu', 'linear'],
                                             learning_rate = 1e-3,
                                             huber_delta = 1.0,
                                             training_epochs = actor_training_epochs,
                                             steps_per_epoch = actor_steps_per_epoch)
        self.critic_target_network = ddpg_neural_network(layersizes = [sl+al, 128, 64, 1],
                                             activations = ['relu', 'relu', 'linear'],
                                             learning_rate = 1e-3,
                                             huber_delta = 1.0,
                                             training_epochs = actor_training_epochs,
                                             steps_per_epoch = actor_steps_per_epoch)
        self.critic_target_network.model.set_weights(self.critic_network.model.get_weights())
        
    
    def act(self, state, ornstein_uhlenbeck):
        if ornstein_uhlenbeck:
            N = self.ornstein_uhlenbeck_process.simulate()
            return self.actor_network.predict_network(np.array(state).reshape((1, -1))).reshape((-1,))[0] + N
        else:
            return self.actor_network.predict_network(np.array(state).reshape((1, -1))).reshape((-1,))[0]
    
    
    def learn(self, next_state, next_state_terminal, next_reward, last_action):

        # Apply Gradient Algorithm if start of a new episode
        if self.last_state_terminal and len(self.episode_queue) > 1:
            
            self.last_state_terminal = False
            for i in np.arange(len(self.episode_queue)-1):
                
                critic_labels = []
                e = self.episode_queue[i]
                for t in np.arange(1, len(e.rewards) - 1):

                    # Set Current Timestep Variables
                    action = np.array([e.actions[t]]).reshape(-1,)
                    state = np.array([e.states[t-1]]).reshape(-1,)
                    next_state = np.array([e.states[t]]).reshape(-1,)
                    rewards = e.rewards[t]
                    
                    # Evaluate Neural Networks
                    predicted_action = self.actor_network.predict_network(np.array(state).reshape((1, -1))).reshape((-1,))
                    action_value = self.critic_network.predict_network(np.concatenate((state, action)).reshape((1, -1))).reshape((-1,))
                    next_target_action = self.actor_target_network.predict_network(np.array(next_state).reshape((1, -1))).reshape((-1,))
                    next_target_action_value = self.critic_target_network.predict_network(np.concatenate((next_state, next_target_action)).reshape((1, -1))).reshape((-1,))
                    
                    # Create a set of critic network labels from training
                    critic_labels.append(rewards + self.df * next_target_action_value)
            
                # Backpropagate Critic Network and Actor Network
                T = len(e.rewards) - 2
                aX = np.array(e.states[0:(len(e.rewards)-2)]).reshape((T,-1))
                cX = np.concatenate((aX, np.array(e.actions[1:(len(e.rewards)-1)]).reshape((T,-1))), axis=1)
                cY = np.array(critic_labels).reshape((T,-1))
                self.critic_network.train_network_as_critic(cX, cY)
                self.actor_network.train_network_as_actor(self.critic_network, aX, T, self.sl, self.al)
                
                
                    
            # Update Actor Target Network
            actor_weights = self.actor_network.model.get_weights()
            actor_target_weights = self.actor_target_network.model.get_weights()
            for i in np.arange(len(actor_weights)): actor_target_weights[i] = self.target_frac*actor_weights[i] + (1-self.target_frac)*actor_target_weights[i]
            self.actor_target_network.model.set_weights(actor_target_weights)
            
            # Update Critic Target Network
            critic_weights = self.critic_network.model.get_weights()
            critic_target_weights = self.critic_target_network.model.get_weights()
            for i in np.arange(len(critic_weights)): critic_target_weights[i] = self.target_frac*critic_weights[i] + (1-self.target_frac)*critic_target_weights[i]
            self.critic_target_network.model.set_weights(critic_target_weights)
                
        # Append episode queues with information from the current State / Action / Reward
        self.episode_queue[len(self.episode_queue) - 1].states.append(next_state)
        self.episode_queue[len(self.episode_queue) - 1].rewards.append(next_reward)
        self.episode_queue[len(self.episode_queue) - 1].actions.append(last_action)

        # Create a new episode if the current episode has just ended
        if next_state_terminal:
            self.last_state_terminal = True
            self.ornstein_uhlenbeck_process.value = 0
            self.episode_queue[len(self.episode_queue) - 1].total_rewards = sum(self.episode_queue[len(self.episode_queue) - 1].rewards)
            self.episode_queue.append(episode())
            while len(self.episode_queue) > self.eql: self.episode_queue.pop(0)


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