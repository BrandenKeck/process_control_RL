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
my_controller = rl_controller(
    pv0 = 22.5,
    out0 = 50,
    sps=25*np.ones(2000),
    lsl = 20,
    usl = 30,
    #pvf=cooling_valve,
    lr=0.0000001,
    df=1,
    eql=6, sl=20,
    ql = 1000
)

#my_controller = rl_controller()
#my_controller.explore(200, exp_factor=0.01)
my_controller.train(200)
my_controller.run()