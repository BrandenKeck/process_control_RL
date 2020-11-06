import numpy as np
from process_control import rl_controller

def cooling_valve(pv, out):
    return pv - (out/100) + 0.25 + np.random.normal(0, 0.01)


setpoints = np.ones(200)
flipper = 1
for i in np.arange(4):
    flipper = flipper - 2*flipper
    setpoints = np.concatenate((setpoints, flipper*np.ones(200)))

#setpoints = np.arange(500)/20 
my_controller = rl_controller(
    pv0 = 22.5,
    out0 = 50,
    sps=setpoints,
    rwd_baseline = 100,
    max_err = 0.1,
    max_err_rwd = 1000,
    #pvf=cooling_valve,
    lr=0.0000000001,
    df=1,
    eql=11, sl=10,
    ql = 500
)

#my_controller.explore(10, exp_factor=0.01)
#my_controller.train(20)
my_controller.run(learn=False)