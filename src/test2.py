import gym
import time

env = gym.make('LunarLander-v2')
env.reset()

for _ in range(1000):
    env.render()
    time.sleep(0.1)
    env.step(env.action_space.sample())  # take a random action

env.close()
