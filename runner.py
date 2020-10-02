import numpy as np
from process_control import rl_controller

def cooling_valve(pv, out):
    return pv - (out/100) + 0.25 #+ np.random.normal(0, 0.01)


setpoints = np.ones(200)
flipper = 1
for i in np.arange(4):
    flipper = flipper - 2*flipper
    setpoints = np.concatenate((setpoints, flipper*np.ones(200)))

#setpoints = np.arange(500)/20 
my_controller = rl_controller(pv0 = 22.5,
    sps=25*np.ones(2000),
    #pvf=cooling_valve,
    lr=0.02e-5,
    df=0.1,
    eql=4, sl=15,
    ql = 200)

#my_controller = rl_controller()
#my_controller.explore(10, exp_factor=0.01)
my_controller.train(1000)
my_controller.run()