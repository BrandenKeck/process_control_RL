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
my_controller = rl_controller(pv0 = 22.5, sps=25*np.ones(800), lr_mean=1e-8, df=0.85, eql=2, sl=50, ql = 800)

#my_controller = rl_controller()
my_controller.explore(10, amplitude=40, period=400)
my_controller.train(10)
my_controller.run()