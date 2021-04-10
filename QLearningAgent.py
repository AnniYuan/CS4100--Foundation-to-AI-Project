import numpy as np

class QLearningAgent:
    def __init__(self,env, epsilon,gamma, lr):
        self.epsilon = epsilon
        self.gamma = gamma
        self.lr_rate = lr
        self.env = env
        self.Q = np.zeros((env.observation_space.n, env.action_space.n)) #16x4

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.Q[state, :])
        return action

    def update(self, state, state2, reward, action):
        predict = self.Q[state, action]
        target = reward + self.gamma * np.max(self.Q[state2, :])
        self.Q[state, action] = self.Q[state, action] + self.lr_rate * (target - predict)


    def choose_play_action(self,state):
        action = np.argmax(self.Q[state, :])
        return action