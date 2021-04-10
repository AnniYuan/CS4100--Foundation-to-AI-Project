import gym
import plot
from QLearningAgent import QLearningAgent
from DynaQAgent import DynaQAgent
from DeepQLearningAgent import DeepQLearningAgent

env = gym.make('FrozenLake8x8-v0')
env.reset()

# env.reset()
# env.render()
epsilon = 0.5
lr_rate = 0.3
gamma = 0.96
step = 6
total_episodes = 1000
max_steps = 1000

qLearning = QLearningAgent(env, epsilon, gamma, lr_rate)
dynaq = DynaQAgent(env, epsilon, gamma, lr_rate, step)
dQ = DeepQLearningAgent(env, epsilon, gamma, lr_rate)


def train(agent):
    win = 0
    scores = []
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
                break
        if s >= 1: win += 1
        if ep % 10 == 0: print('Episode', ep, 'Takes', t, 'steps', 'score:', s)
        scores.append(s)
    print('Win: ', win)
    return scores


def play(agent, numberEpisode=1000):
    win = 0
    for episode in range(numberEpisode):
        state = env.reset()
        t = 0
        score = 0
        while t < 100:
            # env.render()
            action = agent.choose_play_action(state)
            next_state, reward, done, info = env.step(action)
            state = next_state
            t += 1
            score += reward
            if done:
                break
        if score >= 1:
            win += 1
    print('Win: ', win)


#
# train(qLearning)
# play(qLearning,1000)
# train(dynaq)
# play(dynaq,1000)
win = train(dQ)
play(dQ, 1000)
plot.plot_rwd_by_epd(win)
#
# with open("frozenLake_qTable.pkl", 'wb') as f:
#     pickle.dump(QLearningAgent.Q, f)
#
# with open("frozenLake_qTable.pkl", 'rb') as f:
# 	QLearningAgent.Q = pickle.load(f)

# print(QLearningAgent.Q)
