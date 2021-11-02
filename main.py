from Lanechange import LaneChangeEnv
env = LaneChangeEnv('data/cubed.csv','data/cubed_below.csv',50,50,10,20,5, 5, 0.3)
env.reset()
i = 0
while(1):
    env.step(1)
    env.render()
    i += 1
