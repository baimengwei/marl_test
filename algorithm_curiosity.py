import numpy as np
import torch
from torch import nn
import gym
from clean_env import clean_env2
torch.set_default_dtype(torch.float64)


class memory:
    def __init__(self, env, memory_length=20000, memory_minibatch=32):
        self.env = env
        self.memory_length = memory_length
        self.memory_minibatch = memory_minibatch

        self.state_size = self.env.observation_space.shape[0]
        try:
            self.action_size = self.env.action_space.shape[0]
        except:
            self.action_size = 1

        self.memory_width = self.state_size * 2 + self.action_size + 1
        self.memory = np.zeros((self.memory_length, self.memory_width))

        self.index = 0
        self.max_index = 0

    def store(self, state, action, reward, next_state):
        transacton = np.hstack((state, action, reward, next_state))
        self.memory[self.index, :] = transacton

        self.index += 1
        if self.index % self.memory_length == 0:
            self.index = 0
        if self.max_index < self.memory_length:
            self.max_index += 1

    def sample(self):
        choice_random = np.random.choice(self.max_index, self.memory_minibatch)
        choice_data = self.memory[choice_random, :]
        state = choice_data[:, 0:self.state_size]
        action = choice_data[:,
                             self.state_size:self.state_size + self.action_size]
        reward = choice_data[:, self.state_size + self.action_size:
                             self.state_size + self.action_size + 1]
        next_state = choice_data[:, self.state_size + self.action_size + 1:]

        reward = np.squeeze(reward)
        if self.action_size == 1:
            action = np.squeeze(action)
        return state, action, reward, next_state


class curiosity_net(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=32):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = 1  # action_dim
        self.hidden_dim = hidden_dim

        self.linear1 = nn.Linear(self.state_dim, self.hidden_dim)
        self.linear2 = nn.Linear(self.action_dim, self.hidden_dim)
        self.activate = nn.ReLU()
        self.linear3 = nn.Linear(self.hidden_dim, self.state_dim)

        for layer in [self.linear1, self.linear2, self.linear3]:
            torch.nn.init.normal_(layer.weight, mean=0.0, std=0.1)
            torch.nn.init.constant_(layer.bias, 0.0)
        pass

    def forward(self, s, a):
        x1 = self.linear1(s)
        x2 = self.linear2(a)
        x = x1 + x2
        x = self.activate(x)
        x = self.linear3(x)
        return x


class network:
    def __init__(self, env, hidden_dimension=100, learning_rate=1e-3):
        self.env = env
        self.hidden_dimension = hidden_dimension
        self.learning_rate = learning_rate

        self.input_dimension = self.env.observation_space.shape[0]
        self.output_dimension = self.env.action_space.n

        self.model = self.__create_network()
        self.model_curiosity = curiosity_net(
            self.input_dimension, self.output_dimension)

        self.loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.learning_rate)
        self.optimizer_curiosity = torch.optim.Adam(self.model_curiosity.parameters(),
                                                    lr=self.learning_rate)

    @staticmethod
    def replace(network_from, network_to):
        network_to.load_state_dict(network_from.state_dict())

    @staticmethod
    def optimizer(predict_object, predict_value, target_value):
        loss = predict_object.loss(predict_value, target_value)
        predict_object.optimizer.zero_grad()
        loss.backward()
        predict_object.optimizer.step()
        return loss.item()

    def __call__(self, state):
        action_value = self.model(state)
        return action_value

    class q_network_model(nn.Module):
        def __init__(self, in_dim, out_dim, hidden_dim):
            super().__init__()
            self.in_dim = in_dim
            self.out_dim = out_dim
            self.hidden_dim = hidden_dim

            self.linear1 = nn.Linear(self.in_dim, self.hidden_dim)
            self.linear2 = nn.Linear(self.hidden_dim, self.out_dim)
            self.activate = nn.ReLU()
            for layer in [self.linear1, self.linear2]:
                torch.nn.init.normal_(layer.weight, mean=0.0, std=0.1)
                torch.nn.init.constant_(layer.bias, 0.0)

        def forward(self, s):
            x = self.linear1(s)
            x = self.activate(x)
            x = self.linear2(x)
            return x

    def __create_network(self):
        return self.q_network_model(self.input_dimension,
                                    self.output_dimension, self.hidden_dimension)


class agent_q:
    def __init__(self, env, epislon_method=1, gamma=0.98):
        self.env = env
        self.epislon_method = epislon_method
        self.gamma = gamma

        if self.epislon_method == 1:
            self.epislon_method = self.epislon_method_1()

        self.q_network = network(self.env)
        self.q_network_target = network(self.env)
        self.memory = memory(self.env)

        self.epislon_learn_step = 0
        pass

    def output_action(self, state):
        state = torch.from_numpy(state)
        action_value = self.q_network(state)
        action_value = np.array(action_value.tolist())

        random_number = np.random.random()
        if random_number > self.epislon_method.epislon_init:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(action_value)
            action = np.squeeze(action)
        return action, action_value

    def sample_postprocess(self):
        state, action, reward, next_state = self.memory.sample()
        state = torch.from_numpy(state)
        next_state = torch.from_numpy(next_state)
        action = np.squeeze(action)
        reward = np.squeeze(reward)
        action = action.astype(np.int32)
        return state, action, reward, next_state

    def learn(self):
        self.epislon_learn_step += 1

        state, action, reward, next_state = self.sample_postprocess()
        # curiosity
        next_state_fit = self.q_network.model_curiosity(
            state, torch.Tensor(action[np.newaxis].T))
        reward += torch.sum(torch.pow(next_state_fit -
                                      next_state, 2), axis=1).detach().numpy()

        target_value_max, target_action_max = torch.max(
            self.q_network_target.model(next_state), axis=1)
        target_value = reward + self.gamma * \
            np.array(target_value_max.tolist())

        predict_value_all = self.q_network.model(state)

        replace_index = np.arange(self.memory.memory_minibatch, dtype=np.int32)
        target_value_all = np.array(predict_value_all.tolist())
        target_value_all[replace_index, action] = target_value

        self.epislon_method.update()

        if self.epislon_learn_step % 500 == 0:
            state, action, reward, next_state = self.sample_postprocess()
            next_state_fit = self.q_network.model_curiosity(
                state, torch.Tensor(action[np.newaxis].T))
            loss = nn.functional.mse_loss(next_state_fit, next_state)

            self.q_network.optimizer_curiosity.zero_grad()
            loss.backward()
            # print('loss item:', loss.item())
            self.q_network.optimizer_curiosity.step()

        if self.epislon_learn_step % 1000 == 0:
            network.replace(self.q_network.model,
                            self.q_network_target.model)

        return network.optimizer(self.q_network, predict_value_all,
                                 torch.from_numpy(target_value_all))

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
            self.epislon_increment = 1.001
            self.epislon_max = 0.9

        def update(self):
            if self.epislon_init < self.epislon_max:
                self.epislon_init *= self.epislon_increment

    class epislon_method_2:
        def __init__(self):
            self.epislon_init = 0.9

        def update(self):
            pass


class interactive:
    def __init__(self, env, epoch_max=1000, epoch_replace=1):
        self.env = env
        self.epoch_max = epoch_max
        self.epoch_replace = epoch_replace

        self.env = self.env.unwrapped
        self.agent = agent_q(self.env)

    def start_execute(self):
        self.epoch_index = 0
        self.loss_value = 0
        self.render = False
        for i in range(self.epoch_max):
            self.epoch_index += 1
            state = self.env.reset()
            self.epoch_step = 0
            self.reward_total = 0
            while True:
                self.epoch_step += 1
                if self.render:
                    self.env.render()
                action = []
                for i in range(self.env.env.N_agent):
                    a, _ = self.agent.output_action(state)
                    action.append(a)
                next_state, reward, done, info = self.env.step(action)
                self.reward_total += reward

                self.agent.memory.store(state, action, reward, next_state)

                state = next_state

                if self.epoch_index > 1:
                    self.loss_value = self.agent.learn()

                if done:
                    break

            self.statistic()

        self.agent.save_model()

    def statistic(self):
        if not self.epoch_index > 1:
            self.epoch_step_list = []
            self.loss_value_list = []
        else:
            print('epoch %-5s, length %-5s, loss_value %5f, epislon %5f, reward %4d' %
                  (self.epoch_index, self.epoch_step, self.loss_value,
                   self.agent.epislon_method.epislon_init,
                   self.reward_total))
            self.epoch_step_list.append(self.epoch_step)
            self.loss_value_list.append(self.loss_value)


if __name__ == '__main__':
    size = 11
    agent = 1
    max_iter = 5000
    env = clean_env2(agent=agent,
                     max_iter=max_iter, shape=(size, size, 3))

    dqn_evoluate = interactive(env, epoch_max=300)
    dqn_evoluate.start_execute()
