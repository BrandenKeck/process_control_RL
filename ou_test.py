import numpy as np
import matplotlib.pyplot as plt

# Define an Ornstein-Uhlenbeck process for DDPG method exploration
class ornstein_uhlenbeck():
    
    def __init__(self, theta, sigma):
        self.theta = theta
        self.sigma = sigma
        self.value = 0
        
    def simulate(self, dt=1):
        self.value = self.value + self.theta*self.value*dt + self.sigma*np.sqrt(dt)*np.random.normal()
        return self.value

iterations = 150
timesteps = 8000
for i in np.arange(iterations):
    ou = ornstein_uhlenbeck(0, 0.8)
    T = []
    X = []
    for t in np.arange(timesteps):
        ou.simulate()
        T.append(t)
        X.append(ou.value)
    plt.plot(T, X)
    
plt.show()