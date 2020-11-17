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
        
        # Create Neural Networks
        self.actor_network = ddpg_neural_network(layersizes = [sl, 128, 64, 1],
                                             activations = ['relu', 'relu', 'sigmoid'],
                                             learning_rate = 1e-3,
                                             training_epochs = critic_training_epochs,
                                             steps_per_epoch = critic_steps_per_epoch)
        self.actor_target_network = deepcopy(self.actor_network)
        self.critic_network = ddpg_neural_network(layersizes = [sl, 128, 64, 1],
                                             activations = ['relu', 'relu', 'linear'],
                                             learning_rate = 1e-3,
                                             huber_delta = 1.0,
                                             training_epochs = critic_training_epochs,
                                             steps_per_epoch = critic_steps_per_epoch)
        self.critic_target_network = deepcopy(self.critic_network)
        
    
    def act(self, state, ornstein_uhlenbeck):
        if ornstein_uhlenbeck:
            N = self.ornstein_uhlenbeck_process.simulate()
            return self.actor_network.evaluate(np.array(state)) + N
        else:
            return self.actor_network.evaluate(np.array(state))
    
    
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
                    print("State: " + str(state[state.size - 1]))
                    print("Action: " + str(action))
                    print("Value: " + str(action_value))
                    next_target_action = np.array([self.actor_target_network.evaluate(next_state)]).reshape(-1,)
                    next_target_action_value = self.critic_target_network.evaluate(np.concatenate((next_state, next_target_action)))
                    
                    # Calculate Critic Loss and Backpropagate the Actor and Critic Networks
                    print("Target Network Value: " + str(next_target_action_value))
                    print("Rewards: " + str(rewards))
                    print("Loss: " + str(rewards + self.df * next_target_action_value - action_value))
                    #input()
                    delta = rewards + self.df * next_target_action_value - action_value
                    self.critic_network.backpropagate_as_critic(np.concatenate((state, action)), delta)
                    self.actor_network.backpropagate_as_actor(self.critic_network, state, self.sl, self.al)
                    
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