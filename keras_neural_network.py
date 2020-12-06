import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.python.keras.backend as K

class ddpg_neural_network():
    
    def __init__(self, layersizes, activations, 
                 learning_rate=0.01, huber_delta = 1.0,
                 training_epochs = 1, steps_per_epoch = None):
        
        # Store Training Information
        self.training_epochs = training_epochs
        self.steps_per_epoch = steps_per_epoch
        
        # Establish a Model
        self.model = tf.keras.Sequential([keras.Input(shape=(layersizes[0],))])
        for i in np.arange(1, len(layersizes)):
            self.model.add(keras.layers.Dense(units=layersizes[i], activation=activations[i-1]))
        
        # Compile the model with Adam optimizer and huber loss
        self.model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate), 
              loss=tf.losses.Huber(delta=huber_delta),
              metrics=['accuracy'])
    
    # Train the Network from Critic Perspective
    def train_network_as_critic(self, X, Y):
        dataset = tf.data.Dataset.from_tensor_slices((X, Y)).shuffle(len(Y)).batch(len(Y))
        self.model.fit(
            dataset, 
            epochs=self.training_epochs, 
            steps_per_epoch=self.steps_per_epoch,
            verbose=False
        )
    
    # Train the Network from the Actor Perspective     
    def train_network_as_actor(self, critic_network, aX, T, sl, al):
        print(aX.shape)
        with tf.GradientTape() as actor_grad:
            
            actor_input = tf.constant(aX)
            print(actor_input)
            actor_output = self.model.call(actor_input)
            print(actor_output)
            aOut = actor_output.numpy().reshape((T,-1))
            cX = np.concatenate((aX, aOut), axis=1)
            
        actor_gradients = actor_grad.gradient(actor_output, actor_input)
        print(actor_gradients)
        input()
        
        with tf.GradientTape() as critic_grad:
        
            #critic_input = tf.data.Dataset.from_tensor_slices(cX).batch(len(cX))
            critic_input = tf.constant(cX)
            print(critic_input)
            critic_output = critic_network.model.call(critic_input)
            print(critic_output)
        
        critic_gradients = critic_grad.gradient(critic_output, critic_input)
        print(critic_gradients)
        
        
        '''
        # Setup Critic Gradients
        session = K.get_session()
        gradients = K.gradients(critic_network.model.output, critic_network.model.input)
        critic_gradients = session.run(gradients[0], feed_dict={critic_network.model.input: cX}).reshape(-1,)
        critic_action_gradients = critic_gradients[sl:sl+al]
        print(critic_action_gradients)
        
        # Setup Actor Gradients
        gradients = K.gradients(self.model.output, self.model.trainable_weights)
        actor_gradients = session.run(gradients[0], feed_dict={self.model.input: aX})
        print(actor_gradients.shape)
        '''
        input()

    # Predict Network Outputs
    def predict_network(self, X):
        dataset = tf.data.Dataset.from_tensor_slices(X).batch(len(X))
        return self.model.predict(dataset)