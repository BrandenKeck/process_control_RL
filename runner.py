import numpy as np
from process_control import rl_controller

def cooling_valve(pv, out):
    return pv - (out/100) + 1 + np.random.normal(0, 0.01)

flipper = 1
setpoints = flipper*np.ones(200)
for i in np.arange(4):
    flipper = flipper - 2*flipper
    setpoints = np.concatenate((setpoints, flipper*np.ones(200)))

#setpoints = np.arange(500)/20 
my_controller = rl_controller(
    pv0 = 2,
    out0 = 50,
    sps=setpoints,
    rwd_baseline = 0,
    max_err = 0.01,
    max_err_rwd = 1,
    #pvf=cooling_valve,
    lr=1e-8,
    df=1,
    eql=4, sl=10,
    ql = 300
)

#my_controller = rl_controller()
#my_controller.explore(10, exp_factor=0.01)
my_controller.train(10)
my_controller.run()