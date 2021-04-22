from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random
import numpy as np
from Memory import *
import gym
import numpy as np
import time, pickle, os
env = gym.make('FrozenLake8x8-v0')
env.reset()
state_size = 1
action_size = env.action_space.n
print('action space',env.action_space)
print('state space',env.observation_space)
epsilon = 0.9
lr_rate = 0.1
gamma = 0.9
max_steps = 100

class DQNAgent:
    def __init__(self, state_size, action_size, gamma= gamma, epsilon = epsilon, lr_rate = lr_rate):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.dqn_learning_rate = lr_rate
        self.model = self._build_model()
        self.memory = Memory(1000)  # PER Memory
        self.batch_size = 32

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(67, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.dqn_learning_rate))
        return model

    def memorize(self, state, action, reward, next_state, done):
        # Calculate TD-Error for Prioritized Experience Replay
        td_error = reward + self.gamma * np.argmax(self.model.predict(next_state)[0]) - np.argmax(
            self.model.predict(state)[0])
        # Save TD-Error into Memory
        self.memory.add(td_error, (state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:  # Exploration
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action (Exploitation)

    def choose_play_action(self, state):
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action (Exploitation)

    def replay(self):
        batch, idxs, is_weight = self.memory.sample(self.batch_size)
        for i in range(self.batch_size):
            state, action, reward, next_state, done = batch[i]
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            else:
                target = reward
            target_f = self.model.predict(state)
            target_f[0][action] = target
            # Gradient Update. Pay attention at the sample weight as proposed by the PER Paper
            self.model.fit(state, target_f, epochs=1, verbose=0, sample_weight=np.array([is_weight[i]]))
        if self.epsilon > 0.001:  # Epsilon Update
            self.epsilon *= self.epsilon_decay

agent = DQNAgent(state_size, action_size)
scores=[]
EPISODES = 21000
y_axis=[]
def train():
    win = 0
    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        t = 0
        s = 0
        while t < max_steps:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            #reward = reward if not done else -1
            next_state = np.reshape(next_state, [1, state_size])
            agent.memorize(state, action, reward, next_state, done)
            state = next_state
            t += 1
            s += reward
            if done:
                if s >= 1: win += 1
                if EPISODES%100:
                    print("episode: {}/{}, Win: {} ".format(e, EPISODES, win))
                #print("Beta {:.5f} / Eps: {:.5f}".format(agent.memory.beta, agent.epsilon))
                break
        if agent.memory.tree.n_entries > 32:
            agent.replay()
    print(EPISODES,'Episodes, Win:',win)
    y_axis.append(win)


train()

print(y_axis)