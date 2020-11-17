import numpy as np
import tensorflow as tf
from tensorflow import keras

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
    
    # Train the Network        
    def train_network_as_critic(self, X, Y):
        dataset = tf.data.Dataset.from_tensor_slices((X, Y)).shuffle(len(Y)).batch(len(Y))
        self.model.fit(
            dataset, 
            epochs=self.training_epochs, 
            steps_per_epoch=self.steps_per_epoch,
            verbose=False
        )
    
    # Predict Network Outputs
    def predict_network(self, X, Y):
        dataset = tf.data.Dataset.from_tensor_slices((X, Y)).batch(len(Y))
        return self.model.predict(dataset)