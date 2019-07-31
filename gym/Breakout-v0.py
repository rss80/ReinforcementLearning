import gym

env = gym.make('Breakout-v0')
#env.monitor.start('/tmp/Breakout-experiment')

for i_episode in range(20):
    score = 0			# keeps track of the reward
    env.mode='human'
    observation = env.reset()
    for t in range(200):
        env.render()		# starting the environment display screen
#       print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)	
	score += reward	
	        
        if done:
            print("Episode finished after {} timesteps".format(t+1))	    	  
            break

    print("Score :")
    print(str(score))

#env.monitor.close()
