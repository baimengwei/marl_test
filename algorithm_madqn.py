import torch
import gym
import numpy as np
import torch.nn as nn
from clean_env import clean_env2
import matplotlib.pyplot as plt
from multiprocessing import *
torch.set_default_dtype(torch.float64)


class sum_tree:
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.memory_node = np.zeros(memory_size * 2 - 1)
        self.memory_data = np.zeros(memory_size, dtype=object)

        self.priority_upper = 1

        self.memory_pointer = 0

    def add(self, transaction, priority_value):
        self.memory_data[self.memory_pointer] = transaction
        self.memory_pointer += 1
        if self.memory_pointer >= self.memory_size:
            self.memory_pointer = 0

        memory_index = self.memory_size - 1 + self.memory_pointer
        self.update_index(priority_value, memory_index)

    def update_index(self, priority_value, memory_index):
        change_value = priority_value - self.memory_node[memory_index]
        self.memory_node[memory_index] = priority_value
        while memory_index != 0:
            memory_index = (memory_index - 1) // 2
            self.memory_node[memory_index] += change_value

    def get(self, priority_choose):
        node_start = 0
        # import pdb; pdb.set_trace()
        while True:
            node_left = node_start * 2 + 1
            node_right = node_left + 1
            if node_left >= self.memory_size * 2 - 1:
                node_end = node_start
                break
            else:
                if priority_choose <= self.memory_node[node_left]:
                    node_start = node_left
                else:
                    priority_choose -= self.memory_node[node_left]
                    node_start = node_right
        data_index = node_end - (self.memory_size - 1)
        return node_end, self.memory_node[node_end], self.memory_data[data_index]

    @property
    def priority_max(self):
        priority_value = np.max(self.memory_node[-self.memory_size:])
        if priority_value <= 0:
            priority_value = self.priority_upper
        return priority_value

    @property
    def priority_min(self):
        priority_value = np.min(self.memory_node[-self.memory_size:])
        return priority_value

    @property
    def priority_sum(self):
        return self.memory_node[0]


class memory:
    def __init__(self, env, memory_length=20000, memory_minibatch=32, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.env = env
        self.memory_length = memory_length
        self.memory_minibatch = memory_minibatch
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment

        self.state_size = self.env.observation_space.shape[0]
        try:
            self.action_size = self.env.action_space.shape[0]
        except:
            self.action_size = 1
        self.memory_width = self.state_size * 2 + self.action_size + 1

        self.memory = sum_tree(self.memory_length)

    def store(self, state, action, reward, next_state):
        state = state.flatten()
        next_state = next_state.flatten()
        transacton = np.hstack((state, action, reward, next_state))
        priority_value = self.memory.priority_max
        self.memory.add(transacton, priority_value)

    def sample(self):
        # import pdb; pdb.set_trace()
        priority_index_array = np.empty(self.memory_minibatch)
        memory_array = np.empty((self.memory_minibatch, self.memory_width))
        weight_array = np.empty(self.memory_minibatch)

        priority_segment = self.memory.priority_sum / self.memory_minibatch
        if self.beta < 1:
            self.beta += self.beta_increment
        priority_min = self.memory.priority_min / self.memory.priority_sum
        priority_min += 0.0001
        for i in range(self.memory_minibatch):
            left, right = priority_segment * i, priority_segment * (i + 1)
            priority_choose = np.random.uniform(left, right)
            priority_index, priority_value, memory_data = self.memory.get(
                priority_choose)
            weight = np.power(priority_value / priority_min, -self.beta)

            priority_index_array[i] = priority_index
            weight_array[i] = weight
            memory_array[i, :] = memory_data
        return priority_index_array, weight_array, memory_array

    def batch_update(self, priority_index, priority_value):
        priority_value = priority_value + 0.00001
        store_value = np.power(priority_value, self.alpha)
        for index, value in zip(priority_index, store_value):
            value = min(value, self.memory.priority_upper)
            index = int(index)
            self.memory.update_index(value, index)


class cnn_network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 2, 2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(2, 8, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(128, 128)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        # for i in range(3):
            # plt.figure()
            # plt.imshow(x[0][i].detach().numpy())
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        # x = self.pool2(x)
        x = self.flatten(x)

        x = self.activate(self.fc1(x))
        x = self.fc2(x)
        return x


class network:
    def __init__(self, env, hidden_dimension=100, learning_rate=1e-3):
        self.env = env
        self.hidden_dimension = hidden_dimension
        self.learning_rate = learning_rate
        self.input_dimension = self.env.observation_space.shape[0]
        self.output_dimension = self.env.action_space.n

        self.model = self.__create_network()
        self.loss_mse = torch.nn.MSELoss(reduction='none')
        self.loss_l1 = torch.nn.L1Loss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate)

    @staticmethod
    def replace(network_from, network_to):
        network_to.load_state_dict(network_from.state_dict())

    @staticmethod
    def optimizer(predict_object, predict_value, target_value, weight):
        loss = predict_object.loss_mse(predict_value, target_value)
        loss *= torch.Tensor(weight[np.newaxis].T)
        loss = torch.mean(loss)
        predict_object.optimizer.zero_grad()
        loss.backward()
        predict_object.optimizer.step()
        return loss.item()

    def __call__(self, state):
        action_value = self.model(state)
        return action_value

    def __create_network(self):
        return cnn_network()


class agent_q:
    def __init__(self, env, epislon_method=1, gamma=0.99):
        self.env = env
        self.epislon_method = epislon_method
        self.gamma = gamma

        if self.epislon_method == 1:
            self.epislon_method = self.epislon_method_1()

        self.q_network = network(self.env)
        self.q_network_target = network(self.env)
        self.memory = memory(self.env)
        pass

    def output_action(self, state):
        # plt.figure()
        # plt.imshow(state)
        state_in = np.array([state[:, :, 0], state[:, :, 1], state[:, :, 2]])
        state_in = state_in[np.newaxis]
        state = torch.from_numpy(state_in)

        action_value = self.q_network(state)
        action_value = np.array(action_value.tolist())

        random_number = np.random.random()
        if random_number > self.epislon_method.epislon_init:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(action_value)
            action = np.squeeze(action)
        return action, action_value

    def sample_decouple(self, choice_data):
        state = choice_data[:, 0:self.memory.state_size]
        action = choice_data[:, self.memory.state_size:
                             self.memory.state_size + self.memory.action_size]
        reward = choice_data[:, self.memory.state_size + self.memory.action_size:
                             self.memory.state_size + self.memory.action_size + 1]
        next_state = choice_data[:, self.memory.state_size +
                                 self.memory.action_size + 1:]
        return state, action, reward, next_state

    def sample_postprocess(self):
        weight_index, weight, memory_data = self.memory.sample()
        state, action, reward, next_state = self.sample_decouple(memory_data)
        state_in = []
        next_state_in = []
        for i in range(state.shape[0]):
            state_in_each = state[i].reshape(11, 11, 3)
            state_in_each = np.array([state_in_each[:, :, 0],
                                      state_in_each[:, :, 1],
                                      state_in_each[:, :, 2]])
            state_in.append(state_in_each)

            next_state_in_each = next_state[i].reshape(11, 11, 3)
            next_state_in_each = np.array([next_state_in_each[:, :, 0],
                                           next_state_in_each[:, :, 1],
                                           next_state_in_each[:, :, 2]])
            next_state_in.append(next_state_in_each)

        state = np.array(state_in)
        state = torch.from_numpy(state)
        next_state = np.array(next_state_in)
        next_state = torch.from_numpy(next_state)

        action = np.squeeze(action)
        reward = np.squeeze(reward)
        action = action.astype(np.int32)
        return state, action, reward, next_state, weight, weight_index

    def learn(self):
        state, action, reward, next_state, weight, weight_index = self.sample_postprocess()
        target_value_max, target_action_max = torch.max(
            self.q_network_target.model(next_state), axis=1)
        target_value = reward + self.gamma * \
            np.array(target_value_max.tolist())

        predict_value_all = self.q_network.model(state)

        replace_index = np.arange(self.memory.memory_minibatch, dtype=np.int32)
        target_value_all = np.array(predict_value_all.tolist())
        target_value_all[replace_index, action] = target_value

        self.epislon_method.update()

        predict_value = predict_value_all[replace_index, action]
        td_error = target_value - np.array(predict_value.tolist())
        priority_value = np.abs(td_error)
        self.memory.batch_update(weight_index, priority_value)

        return network.optimizer(self.q_network, predict_value_all,
                                 torch.from_numpy(target_value_all),
                                 weight)

    def save_model(self, dir_name='D:\\'):
        torch.save(self.q_network.model, dir_name + 'q_network')
        torch.save(self.q_network.optimizer, dir_name + 'q_network_optimizer')
        torch.save(self.q_network_target.model, dir_name + 'q_target_network')
        torch.save(self.q_network_target.optimizer,
                   dir_name + 'q_network_target_optimizer')

    def load_model(self, dir_name='D:\\'):
        self.q_network.model = torch.load(dir_name + 'q_network')
        self.q_network.optimizer = \
            torch.optim.Adam(self.q_network.model.parameters(),
                             lr=1e-3)
        self.q_network_target.model = torch.load(dir_name + 'q_target_network')
        self.q_network_target.optimizer = \
            torch.optim.Adam(self.q_network.model.parameters(),
                             lr=1e-3)

    class epislon_method_1:
        def __init__(self):
            self.epislon_init = 0.01
            self.epislon_increment = 1.0005
            self.epislon_max = 0.90

        def update(self):
            if self.epislon_init < self.epislon_max:
                self.epislon_init *= self.epislon_increment

    class epislon_method_2:
        def __init__(self):
            self.epislon_init = 0.9

        def update(self):
            pass


class interactive(Process):
    def __init__(self, env, qv_mag=None, barrier=None, agent_name=None,
                 epoch_max=1000, epoch_replace=10):
        Process.__init__(self)
        self.env = env
        self.qv_mag = qv_mag
        self.barrier = barrier
        self.agent_name = agent_name
        self.epoch_max = epoch_max
        self.epoch_replace = epoch_replace

        self.agent = agent_q(self.env['env'])

    def run(self):
        self.epoch_index = 0
        self.loss_value = 0
        for i in range(self.epoch_max):
            self.epoch_index += 1
            state = self.env['env'].reset()
            self.epoch_step = 0
            self.total_reward = 0
            while True:
                if i > 10:
                    self.env.render()
                self.epoch_step += 1

                action = self.agent.output_action(state)
                self.qv_mag[self.agent_name] = action
                self.barrier.wait()

                action = [self.qv_mag[each][0] for each in self.qv_mag.keys()]
                if self.agent_name == 'agent0':
                    next_state, reward, done, info = \
                        self.env['env'].step(action)
                    self.env['env'].set_update_flag(True)
                    self.barrier.wait()
                else:
                    while True:
                        if self.env.update_flag:
                            next_state = self.env.state
                            reward = self.env.reward
                            break
                    self.barrier.wait()
                    self.env.update_flag = False

                self.total_reward += reward

                self.agent.memory.store(state, action, reward, next_state)
                state = next_state

                if self.epoch_index > 1:
                    self.loss_value = self.agent.learn()

                if done:
                    break

            if self.epoch_index % self.epoch_replace == 0:
                network.replace(self.agent.q_network.model,
                                self.agent.q_network_target.model)

            self.statistic()

    def statistic(self):
        if not self.epoch_index > 1:
            self.epoch_step_list = []
            self.loss_value_list = []
        else:
            print('epoch %-5s, reward %-5s, loss_value %10s, epislon %5f, epoch_step %5d' %
                  (self.epoch_index, self.total_reward, self.loss_value,
                   self.agent.epislon_method.epislon_init, self.epoch_step))


class MultiInteractivate:
    def __init__(self):
        size = 11
        agent = 3
        max_iter = 5000
        env = Manager().dict({'env': clean_env2(agent=agent,
                                                max_iter=max_iter,
                                                shape=(size, size, 3))})
        qv_mag = self.create_mp_q_value()
        barrier = Barrier(agent)

        self.proc = [interactive(env, qv_mag, barrier, 'agent' + str(i),
                                 epoch_max=300)
                     for i in range(agent)]

    def run(self):
        for each in self.proc:
            each.start()
        for each in self.proc:
            each.join()

    def create_mp_q_value(self):
        manager = Manager()
        q_value = manager.dict({})
        return q_value


if __name__ == '__main__':

    start_obj = MultiInteractivate()
    start_obj.run()
    # reward_list = test.run()
    # plt.plot(reward_list)
    # plt.show()
