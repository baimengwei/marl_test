import numpy as np
import random
from clean_env import clean_env2
import pandas as pd
import matplotlib.pyplot as plt


class MAQLearning():
    def __init__(self, env, discount=0.8, n_iter=1000):

        self.env = env
        self.S = env.observation_space.n
        self.A = env.action_space.n

        self.agent_num = self.env.env.N_agent

        self.max_iter = int(n_iter)
        self.discount = discount

        self.Q = np.zeros((self.agent_num, self.S * self.S, self.A))

    def run(self):
        reward_list = []

        for n in range(0, self.max_iter):
            s = self.env.reset()
            r_total = 0
            while True:
                if n > 500:
                    self.env.render()
                a_list = []
                for i in range(self.agent_num):
                    pn = np.random.random()
                    s_agent = s[i] + s[(i + 1) % 2] * self.S
                    if pn < 0.95:  # n / self.max_iter * 2:
                        a = self.Q[i][s_agent, :].argmax()
                    else:
                        a = self.env.action_space.sample()
                    a_list.append(a)

                s_new, r, done, _ = self.env.step(a_list)
                r -= 1

                r_total += r
                for i in range(self.agent_num):
                    s_agent = s[i] + s[(i + 1) % 2] * self.S
                    s_agent_new = s_new[i] + s_new[(i + 1) % 2] * self.S
                    delta = r + self.discount * \
                        self.Q[i][s_agent_new, :].max() - \
                        self.Q[i][s_agent, a_list[i]]
                    dQ = 0.5 * delta

                    self.Q[i][s_agent, a_list[i]
                              ] = self.Q[i][s_agent, a_list[i]] + dQ
                # print(s, a_list, r, s_new, done)
                s = s_new

                if done:
                    break
            reward_list.append(r_total)
            print('--------------------------------------')
            print('episode %d, reward: %s' % (n, r_total))

            print('--------------------------------------')
        return reward_list


if __name__ == '__main__':

    size = 9
    agent = 2
    max_iter = 1000
    env = clean_env2(size=size, agent=agent, max_iter=max_iter)

    test = MAQLearning(env)
    reward_list = test.run()

    plt.plot(reward_list)
    plt.show()
