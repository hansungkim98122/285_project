import numpy as np
import time
import gym
import TeachMyAgent_modified.TeachMyAgent.environments

env = gym.make('parametric-continuous-parkour-v0', agent_body_type='classic_bipedal', movable_creepers=True)
input_vector = np.array([-0.058,0.912,0.367])
env.set_environment(input_vector=input_vector, water_level = 0.1)
print(env.world.contactListener)
env.reset()
print(env.world.contactListener)
env.reset()
print(env.world.contactListener)

total_steps =1000

for steps in range(total_steps):
    _, _, d, _ = env.step(env.action_space.sample())
    env.render()
    time.sleep(0.1)
    if steps%100==0:
        print('resetting')
        env.reset()

