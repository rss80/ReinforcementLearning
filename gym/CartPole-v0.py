import gym

env = gym.make('CartPole-v0')
#env.monitor.start('/tmp/cartpole-experiment-1')

for i_episode in range(20):
    score = 0			# keeps track of the reward
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
