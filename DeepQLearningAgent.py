import numpy as np
import tensorflow.compat.v1 as tf

tf.compat.v1.disable_eager_execution()


class DeepQLearningAgent:
    def __init__(self, env, epsilon, gamma, lr):
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.lr_rate = lr
        self.n_obs, self.n_actions = env.observation_space.n, env.action_space.n
        self.x = tf.placeholder(shape=[1, self.n_obs], dtype=tf.float32)
        self.W = tf.Variable(tf.random_uniform([self.n_obs, self.n_actions], 0, 0.1))
        self.out = tf.matmul(self.x, self.W)
        self.act = tf.argmax(self.out, 1)
        self.t = tf.placeholder(shape=[1, self.n_actions], dtype=tf.float32)
        self.loss = tf.reduce_sum(tf.square(self.t - self.out))
        self.train_step = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(self.loss)
        self.q_predict = None
        self.sess = tf.Session()
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

    def one_hot(self, state):
        action, qvalues = self.sess.run([self.act, self.out],
                                        feed_dict={self.x: np.identity(self.n_obs)[state:state + 1]})
        return action, qvalues

    def choose_action(self, state):
        action, qvalues = self.one_hot(state)
        self.q_predict = qvalues
        if np.random.uniform(0, 1) < self.epsilon:
            action[0] = self.env.action_space.sample()
        return action[0]

    def choose_play_action(self, state):
        action, qvalues = self.one_hot(state)
        return action[0]

    def update(self, state, next_state, reward, action):
        qnext_values = self.sess.run([self.out], feed_dict={self.x: np.identity(self.n_obs)[next_state:next_state + 1]})
        targetq = self.q_predict
        targetq[0, action] = reward + self.gamma * np.max(qnext_values)
        self.sess.run([self.train_step],
                      feed_dict={self.x: np.identity(self.n_obs)[state:state + 1], self.t: targetq})
