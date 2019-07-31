import gym
import numpy as np

env = gym.make('CartPole-v0')
# env.monitor.start('/tmp/cartpole-experiment-1')

def run_episode(env, parameters):  
    observation = env.reset()
   # env.render()
    totalreward = 0
    for _ in xrange(200):
        action = 0 if np.matmul(parameters,observation) < 0 else 1
        observation, reward, done, info = env.step(action)
        totalreward += reward
        if done:
            break
    return totalreward

bestparams = None  
bestreward = 0  

for _ in xrange(10000):
  #  observation = env.reset()  
    env.render()

    parameters = np.random.rand(4) * 2 - 1
    reward = run_episode(env,parameters)
    if reward > bestreward:
        bestreward = reward
        bestparams = parameters
        
        if reward == 1000:
            break

