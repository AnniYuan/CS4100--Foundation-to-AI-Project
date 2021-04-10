import numpy as np


class DynaQAgent:
    def __init__(self, env, epsilon, gamma, lr,loop):
        self.steps = loop
        self.model = {}
        self.epsilon = epsilon
        self.gamma = gamma
        self.lr_rate = lr
        self.env = env
        self.Q = np.zeros((env.observation_space.n, env.action_space.n))  # 16x4

    def update(self, state, state2, reward, action):
        #update Q table
        predict = self.Q[state, action]
        target = reward + self.gamma * np.max(self.Q[state2, :])
        self.Q[state, action] = self.Q[state, action] + self.lr_rate * (target - predict)
        #update model
        if state not in self.model.keys():
            self.model[state] = {}
        self.model[state][action] = (reward, state2)
        # loop n times to randomly update Q-value
        for _ in range(self.steps):
            # randomly choose an state
            idx = np.random.choice(range(len(self.model.keys())))
            rstate = list(self.model)[idx]
            # randomly choose an action
            idx2 = np.random.choice(range(len(self.model[rstate].keys())))
            raction = list(self.model[rstate])[idx2]

            rreward, r_next_State = self.model[rstate][raction]

            self.Q[rstate,raction] += self.lr_rate * (
                        rreward + self.gamma * np.max(self.Q[r_next_State, :]) - self.Q[rstate,raction])

    def choose_action(self, state):
            if np.random.uniform(0, 1) < self.epsilon:
                action = self.env.action_space.sample()
            else:
                action = np.argmax(self.Q[state, :])
            return action

    def choose_play_action(self, state):
        action = np.argmax(self.Q[state, :])
        return action
