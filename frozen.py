import gym
from QLearningAgent import QLearningAgent
from DynaQAgent import DynaQAgent
import numpy as np
import time, pickle, os
from DeepQLearningAgent import DeepQLearningAgent
env = gym.make('FrozenLake8x8-v0')
env.reset()

# env.reset()
# env.render()
epsilon = 0.9
lr_rate = 0.1
gamma = 0.9
step = 5
max_steps = 5000
total_episodes = 5000

qLearning = QLearningAgent(env, epsilon, gamma, lr_rate)
dynaq = DynaQAgent(env, epsilon, gamma, lr_rate, step)
dQ = DeepQLearningAgent(env,epsilon,gamma,lr_rate)

def train(agent):
    win = 0
    for ep in range(total_episodes):
        state = env.reset()
        t = 0
        s = 0
        while t < max_steps:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, next_state, reward, action)
            state = next_state
            t += 1
            s += reward
            if done:
                if agent.epsilon > 0.001:
                    agent.epsilon -= 1.0/total_episodes
                break
        if s >= 1 : win += 1
        #if ep % 1000 == 0:print('Episode', ep, 'Takes',t ,'steps','score:', s)
    print ('Train Win: ',win)

def play(agent, numberEpisode=5000):
    win = 0
    for episode in range(numberEpisode):
        state = env.reset()
        t = 0
        score = 0
        while t < 100:
            #env.render()
            action = agent.choose_play_action(state)
            next_state, reward, done, info = env.step(action)
            state = next_state
            t+=1
            score +=reward
            if done:
                break
        if score >= 1:
            win += 1
    print('Episodes: ',total_episodes,'Play Win: ',win)
#


#for _ in range(3):
while total_episodes <= 10000:
    train(dynaq)
    play(dynaq)
    total_episodes+=5000
# train(dynaq)
# play(dynaq,1000)
#train(dQ)
#play(dQ,1000)
#
# with open("frozenLake_qTable.pkl", 'wb') as f:
#     pickle.dump(QLearningAgent.Q, f)
#
# with open("frozenLake_qTable.pkl", 'rb') as f:
# 	QLearningAgent.Q = pickle.load(f)

#print(QLearningAgent.Q)

