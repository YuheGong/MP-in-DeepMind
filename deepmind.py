import numpy as np
import matplotlib.pyplot as plt
from envs.ball_in_cup import BICEnv
from env_wrapper import DMEnvWrapper
import cma





base_env = BICEnv()
env = DMEnvWrapper(base_env, render_size = (480, 480))#, construct_obs_fn = base_env.get_observation)
env.reset()
width = 480
height = 480

dim = 2
params = np.zeros((1, dim))
algo = cma.CMAEvolutionStrategy(x0=params, sigma0=1, inopts={"popsize": 14})

#time_step = env.reset()
for i in range(20):
    action = np.ones(2)
    time_step = env.step(action)
    #env.physics.render()
    env.render()#(height, width, camera_id=1)
    #print(time_step.reward, time_step.discount, time_step.observation)

    #img = plt.imshow(video[i])
    #plt.pause(0.01)  # Need min display time > 0.0.
    #plt.draw()
    video = env.render()#(height, width, camera_id=0)
    # for i in range(max_frame):
    img = plt.imshow(video)
    plt.pause(0.01)  # Need min display time > 0.0.
    plt.draw()