import numpy as np
from process_control import rl_controller

'''
setpoints = np.ones(200)
flipper = 1
for i in np.arange(4):
    flipper = flipper - 2*flipper
    setpoints = np.concatenate((setpoints, flipper*np.ones(200)))
my_controller = rl_controller(sps=setpoints)
'''

my_controller = rl_controller()
my_controller.run_sim()