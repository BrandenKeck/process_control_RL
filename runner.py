import numpy as np
from process_control import rl_controller

def cooling_valve(pv, out):
    return pv - (out/100) + 0.25 #+ np.random.normal(0, 0.01)

setpoints = np.ones(200)
flipper = 1
for i in np.arange(4):
    flipper = flipper - 2*flipper
    setpoints = np.concatenate((setpoints, flipper*np.ones(200)))


#setpoints = np.arange(500)/20 #pvf=cooling_valve,
my_controller = rl_controller(pv0 = 22.5, 
    sps=25*np.ones(1000), 
    lr_mean=1e-6, 
    df=0.85, 
    eql=4, sl=2, 
    tolerance=1, reward_within_tolerance=20, 
    ql = 1000)

#my_controller = rl_controller()
my_controller.explore(50, exp_factor=0.025)
my_controller.train(100)
my_controller.run()