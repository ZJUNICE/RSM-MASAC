# import the experiment-specific parameters from flow.benchmarks
from flow.benchmarks.grid1 import flow_params

# import the make_create_env to register the environment with OpenAI gym
from flow.utils.registry import make_create_env

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random

from torch.distributions import Normal
from copy import deepcopy
from collections import deque


# hyper parameters
agents_num = 25


def clip_grad_norm(parameters, max_norm, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.data.mul_(clip_coef)
    return total_norm


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action_value):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action_value = float(max_action_value)

        # network
        self.fc1 = nn.Linear(self.state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc_pi = nn.Linear(300, self.action_dim)
        self.fc_std = nn.Linear(300, self.action_dim)

    def pi(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_pi(x)) * self.max_action_value
        return mu

    def std(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        scale = F.softplus(self.fc_std(x)) + 1e-5
        return scale

    def select_action(self, state, deterministic):
        with torch.no_grad():
            mu = self.pi(state)
            scale = self.std(state)
            dist = Normal(loc=mu, scale=scale)
            if not deterministic:
                action = dist.sample()
            else:
                action = mu
        return torch.clamp(action, -self.max_action_value, self.max_action_value)

    def dist(self, state):
        mu = self.pi(state)
        scale = self.std(state)
        return Normal(loc=mu, scale=scale)


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.state_dim = state_dim

        # network
        self.fc1 = nn.Linear(self.state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc_v = nn.Linear(300, 1)

    def v(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.fc_v(x)
        return value


class Policy(object):
    def __init__(self,
                 num_inputs,
                 num_outputs,
                 max_action_value,
                 learning_rate=2.5e-4,
                 clip_param=0.2,
                 ppo_epoch=4,
                 gamma=0.99,
                 lmbda=0.95,
                 entropy_coef=0.01,
                 max_grad_norm=0.5,
                 eps=1e-5,
                 buffer_length=500,
                 batch_size=250,
                 device=None):

        # set parameters
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lmbda = lmbda
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.eps = eps
        self.buffer_length = buffer_length
        self.batch_size = batch_size

        self.device = device

        # network
        self.state_dim = num_inputs
        self.action_dim = num_outputs
        self.max_action_value = max_action_value

        self.actor = Actor(self.state_dim, self.action_dim, self.max_action_value).to(self.device)
        self.target_actor = Actor(self.state_dim, self.action_dim, self.max_action_value).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate, eps=self.eps)

        self.critic = Critic(self.state_dim).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate * 2, eps=self.eps)

        # replay buffer
        self.memory = ReplayBuffer(buffer_length=self.buffer_length)

        # collected data
        self.data = []

    def dist(self, state):
        return self.actor.dist(state)

    def target_dist(self, state):
        return self.target_actor.dist(state)

    def v(self, state):
        return self.critic.v(state)

    def select_action(self, state, deterministic):
        return self.actor.select_action(state, deterministic)

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_list, a_list, r_list, s_prime_list, done_list = [], [], [], [], []

        for transition in self.data:
            s, a, r, s_prime, done = transition
            s_list.append(s)
            a_list.append([a])
            r_list.append([r])
            s_prime_list.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_list.append([done_mask])

        s, a, r, s_prime, done_mask = torch.tensor(s_list, dtype=torch.float).to(self.device), \
            torch.tensor(a_list, dtype=torch.float).to(self.device), \
            torch.tensor(r_list, dtype=torch.float).to(self.device), \
            torch.tensor(s_prime_list, dtype=torch.float).to(self.device), \
            torch.tensor(done_list, dtype=torch.float).to(self.device)

        self.data = []

        return s, a, r, s_prime, done_mask

    def train_net(self):
        s, a, r, s_prime, done_mask = self.make_batch()

        with torch.no_grad():
            last_state_value = self.v(s_prime[-1]) * done_mask[-1]

        target_value = deepcopy(r)
        for step in reversed(range(r.size(0))):
            last_state_value = self.gamma * last_state_value + r[step]
            target_value[step] = last_state_value

        for i in range(self.ppo_epoch):
            with torch.no_grad():
                advantage = target_value - self.v(s)
                advantage = torch.tanh(advantage)
            # update actor
            # ppo2
            with torch.no_grad():
                old_dist = self.target_dist(s)
            old_log_prob = old_dist.log_prob(a)
            old_log_prob = torch.clamp(old_log_prob, -20.0, 0.0)

            dist = self.dist(s)
            log_prob = dist.log_prob(a)
            log_prob = torch.clamp(log_prob, -20.0, 0.0)
            ratio = torch.exp(log_prob - old_log_prob)
            ratio = torch.clamp(ratio, 0.0, 10.0)

            entropy = dist.entropy()

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantage
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            clip_grad_norm(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            critic_loss = F.mse_loss(self.v(s), target_value.detach()).mean()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            clip_grad_norm(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()


class ReplayBuffer(object):
    def __init__(self, buffer_length):
        self.buffer = deque(maxlen=buffer_length)

    def put(self, seq_data):
        self.buffer.append(seq_data)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        return mini_batch


def main():
    create_env, env_name = make_create_env(flow_params, version=0)
    env = create_env()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # agent
    num_inputs = 11
    num_outputs = 1
    max_action_value = env.action_space.high[0]
    agents = [Policy(num_inputs=num_inputs, num_outputs=num_outputs, max_action_value=max_action_value,
                     device=device) for _ in range(agents_num)]
    for i in range(agents_num-1):
        agents[i].actor.load_state_dict(agents[-1].actor.state_dict())
        agents[i].critic.load_state_dict(agents[-1].critic.state_dict())

    horizon = 400
    num_steps = 64
    num_epochs = 500

    # recording data
    score = 0.0
    training_score = []

    training_interval = []
    score_interval = []
    print_interval = 10

    # begin simulation
    for n_step in range(num_epochs):
        # initialize environment
        obs = env.reset()
        done = False
        # epoch
        while not done:
            # forward step
            for step in range(num_steps):
                actions = []
                for i in range(agents_num):
                    individual_obs = torch.from_numpy(obs[i]).float().to(device)
                    action_value = agents[i].select_action(individual_obs, deterministic=False)
                    action_value = action_value.detach().cpu().numpy()
                    actions.append(action_value[0])
                obs_prime, reward, done, _ = env.step(actions)
                # save transitions
                for i in range(agents_num):
                    agents[i].put_data((obs[i], actions[i], reward[i], obs_prime[i], done))
                obs = obs_prime
                if done:
                    break

            # A2C and PPO updating
            for i in range(agents_num):
                agents[i].train_net()

        # recording
        if n_step % print_interval == 0 and n_step != 0:
            # test
            for _ in range(1):
                # initialize environment
                obs = env.reset()
                done = False
                # epoch
                while not done:
                    # forward step
                    for step in range(num_steps):
                        actions = []
                        for i in range(agents_num):
                            individual_obs = torch.from_numpy(obs[i]).float().to(device)
                            action_value = agents[i].select_action(individual_obs, deterministic=True)
                            action_value = action_value.detach().cpu().numpy()
                            actions.append(action_value[0])
                        obs_prime, reward, done, _ = env.step(actions)
                        # save transitions
                        obs = obs_prime
                        score += reward[-1] / horizon
                        if done:
                            break

            print('# of episode: {}, avg score: {}'.format(n_step, score / average_times))
            training_score.append(score / average_times)
            training_interval.append(n_step)
            score = 0.0

    # plot
    plt.figure(2)
    plt.plot(training_interval, training_score, 'b.-', linewidth=0.5)
    # plt.axis([1, STEP/TEST_INTERVAL, 0, MAX_TEST_STEP])
    plt.title('Training Performance')
    plt.xlabel('Training Indexes')
    plt.ylabel('Accumulated Reward')
    plt.axis([0, 500, 0, 1])
    plt.show()

    np.save('training_interval.npy', training_interval)
    np.save('my_grid1_indiv.npy', training_score)


if __name__ == "__main__":
    main()
