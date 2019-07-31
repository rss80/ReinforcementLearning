import random, numpy, math, gym

#Neural network


from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *


class NeuralNetowrk:
    def __init__(self, no_of_states, no_of_actions):
        self.no_of_states = no_of_states
        self.no_of_actions = no_of_actions

        self.model = self._createNetwork()
        # self.model.load_weights("cartpole-basic.h5")

    def _createNetwork(self):
        model = Sequential()

        model.add(Dense(output_dim=64, input_dim=no_of_states,  activation='relu'))
        model.add(Dense(output_dim=no_of_actions, activation='linear'))

        opt = RMSprop(lr=0.00025)
        model.compile(loss='mse', optimizer=opt)

        return model

    def train(self, x, y, epoch=1, verbose=0):
        self.model.fit(x, y, batch_size=64, nb_epoch=epoch, verbose=verbose)

    def predict(self, s):
        return self.model.predict(s)

    def predictOne(self, s):
        return self.predict(s.reshape(1, self.no_of_states)).flatten()

#Storing into the memory and taking from memory

class Memory:   # stored as ( s, a, r, s_ )
    samples = []

    def __init__(self, capacity):
        self.capacity = capacity

    def add(self, sample):
        self.samples.append(sample)        

        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)

#agent code
print "Values in bracket indicate ideal values "
MAX_MEMORY = input("Enter max memory (100000) - ")#100000
BATCH_SIZE = input("Enter batch size (64) - ")#64

GAMMA = input("Enter gamma (0.99) - ")#0.99

MAX_EPSILON = input("Enter max epsilon (1) - ")#1
MIN_EPSILON = input("Enter min epsilon (0.01) - ")#0.01
LAMBDA =input("Enter lambda (0.001) - ")# 0.001      # decaying speed

class Agent:
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self, no_of_states, no_of_actions):
        self.no_of_states = no_of_states
        self.no_of_actions = no_of_actions

        self.NeuralNetowrk = NeuralNetowrk(no_of_states, no_of_actions)
        self.memory = Memory(MAX_MEMORY)
        
    def action(self, s):
        if random.random() < self.epsilon:
            return random.randint(0, self.no_of_actions-1)
        else:
            return numpy.argmax(self.NeuralNetowrk.predictOne(s))

    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.add(sample)        

        # decreasing the epsilon
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    def replay(self):    
        batch = self.memory.sample(BATCH_SIZE)
        batchLen = len(batch)

        no_state = numpy.zeros(self.no_of_states)

        states = numpy.array([ o[0] for o in batch ])
        states_ = numpy.array([ (no_state if o[3] is None else o[3]) for o in batch ])

        p = agent.NeuralNetowrk.predict(states)
        p_ = agent.NeuralNetowrk.predict(states_)

        x = numpy.zeros((batchLen, self.no_of_states))
        y = numpy.zeros((batchLen, self.no_of_actions))
        
        for i in range(batchLen):
            o = batch[i]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]
            
            t = p[i]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + GAMMA * numpy.amax(p_[i])

            x[i] = s
            y[i] = t

        self.NeuralNetowrk.train(x, y)

# setting the environment
class Environment:
    def __init__(self, problem):
        self.problem = problem
        self.env = gym.make(problem)

    def run(self, agent):
        s = self.env.reset()
        R = 0 

        while True:            
            self.env.render()

            a = agent.action(s)

            s_, r, done, info = self.env.step(a)

            if done: # if it will be the terminal state
                s_ = None

            agent.observe( (s, a, r, s_) )
            agent.replay()            

            s = s_
            R += r

            if done:
                break

	print "episode no: ",episodes
        print "Total reward:", R
	print " "

#Calling the environmwent...starting part

PROBLEM = 'CartPole-v0'
env = Environment(PROBLEM)

no_of_states  = env.env.observation_space.shape[0]
no_of_actions = env.env.action_space.n

agent = Agent(no_of_states, no_of_actions)
episodes=1

try:
    while True:
        env.run(agent)
	episodes=episodes+1
finally:
    agent.NeuralNetowrk.model.save("cartpole-basicssss.h5")
