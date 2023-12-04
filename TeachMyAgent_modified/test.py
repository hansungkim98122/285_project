import numpy as np
import time
import gym
import TeachMyAgent.environments

env = gym.make('parametric-continuous-parkour-v0', agent_body_type='classic_bipedal', movable_creepers=True)
input_vector = np.ones(3)
env.set_environment(input_vector=input_vector, water_level = 0.1)
env.reset()

while True:
    _, _, d, _ = env.step(env.action_space.sample())
    env.render()
    time.sleep(0.1)
