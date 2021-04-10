# simple neural network implementation of qlearning
import gym
from gym import wrappers
import numpy as np
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()

# build environment
env = gym.make("FrozenLake-v0")
env = wrappers.Monitor(env, '/tmp/frozenlake-qlearning', force=True)
n_obv = env.observation_space.n
n_acts = env.action_space.n

class DeepQLearningAgent:
    def __init__(self,env, epsilon, gamma, lr):
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.lr_rate = lr
        self.x = tf.placeholder(shape=[1, 16], dtype=tf.float32)
        self.W = tf.Variable(tf.random_uniform([16, 4], 0, 0.1))
        self.out = tf.matmul(self.x, self.W)
        self.act = tf.argmax(self.out, 1)
        self.t = tf.placeholder(shape=[1, 4], dtype=tf.float32)
        self.loss = tf.reduce_sum(tf.square(self.t - self.out))
        self.train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self.loss)
        self.q_predict = None
        self.sess = tf.Session()
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

    def choose_action(self, state):
        action, qvalues = self.sess.run([self.act, self.out], feed_dict={self.x: np.identity(16)[state:state + 1]})
        self.q_predict = qvalues
        if np.random.uniform(0, 1) < self.epsilon:
                action[0] = self.env.action_space.sample()
        return action[0]

    def choose_play_action(self, state):
        action, qvalues = self.sess.run([self.act, self.out], feed_dict={self.x: np.identity(16)[state:state + 1]})
        return action[0]

    def update(self, state, next_state, reward, action):
        qnext_values = self.sess.run([self.out], feed_dict={self.x: np.identity(16)[next_state:next_state + 1]})
        targetq = self.q_predict
        targetq[0, action] = reward + gamma * np.max(qnext_values)
        self.sess.run([self.train_step],
                       feed_dict={self.x: np.identity(16)[state:state + 1], self.t: targetq})

# initialization
learning_rate = 0.1
gamma = 0.99
train_episodes = 5000
episodes = 0
prev_state = env.reset()
episode_t = 0
epsilon = 0.1
# create model
# x = tf.placeholder(shape=[1, 64], dtype=tf.float32)
# W = tf.Variable(tf.random_uniform([64, 4], 0, 0.1))
# out = tf.matmul(x, W)
# act = tf.argmax(out, 1)
# t = tf.placeholder(shape=[1, 4], dtype = tf.float32)
# loss = tf.reduce_sum(tf.square(t - out))
# train_step = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss)
#
# # start session
# sess = tf.Session()
# init = tf.global_variables_initializer()
# sess.run(init)

agent = DeepQLearningAgent(env,epsilon,gamma,learning_rate)
# while episodes < train_episodes:
#     episode_t += 1
#     # take noisy action
#     # action, qvalues = sess.run([act, out], feed_dict={x: np.identity(16)[prev_state:prev_state + 1]})
#     # if (np.random.rand(1)) < epsilon:
#     #     action[0] = env.action_space.sample()
#     # next_state, rew, done, _ = env.step(action[0])
#     a =agent.choose_action(prev_state)
#     next_state, rew, done, _ = env.step(a)
#     # find targetQ values and update model
#     agent.update(prev_state,next_state,rew,a)
#     # qnext_values = agent.sess.run([agent.out], feed_dict={agent.x: np.identity(16)[next_state:next_state + 1]})
#     # targetq = agent.q_predict
#     # targetq[0, a] = rew + gamma * np.max(qnext_values)
#     # agent.sess.run([agent.train_step], feed_dict={agent.x: np.identity(16)[prev_state:prev_state + 1], agent.t: targetq})
#     prev_state = next_state
#
#     # episode finished
#     if done:
#         episodes += 1
#         # decrease noise as number of episodes increases
#         epsilon = 1. / ((episodes / 50) + 10)
#         prev_state = env.reset()
#         print ("episode %d finished after %d timesteps, rew = %d" % (episodes, episode_t, rew))
#         episode_t = 0