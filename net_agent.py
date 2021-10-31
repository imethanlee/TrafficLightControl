import torch
import torch.nn as nn
import os
import random
from collections import namedtuple, deque

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, inputs, outputs):
        super(DQN, self).__init__()
        self.shared_layer = nn.Linear(inputs, 20)
        self.seperate0_layer = nn.Linear(20, 20)
        self.seperate1_layer = nn.Linear(20, 20)
        self.out_layer0 = nn.Linear(20, outputs)
        self.out_layer1 = nn.Linear(20, outputs)

    def forward(self, inputs, cur_phase):
        activate = nn.Sigmoid()
        x = activate(self.shared_layer(inputs))
        x_0 = activate(self.seperate0_layer(x))
        x_1 = activate(self.seperate1_layer(x))
        q0_value, q1_value = self.out_layer0(x_0), self.out_layer1(x_1)
        q_value = (1 - cur_phase) * q0_value + cur_phase * q1_value
        return q_value.view(q_value.size(0), -1)


class NetAgent:
    def __init__(self, args):
        self.args = args
        self.num_phases = args.num_phases
        self.num_actions = args.num_actions
        self.memory_size = args.memory_size
        self.memory = self.build_memory()
        self.batch_size = args.batch_size
        # self.device = args.device
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.EPSILON, self.GAMMA = 0.9, 0.9
        self.q_target_outdated = 0
        self.UPDATE_Q_TAR = 5
        self.q_network, self.q_target = DQN(args.state_dim, args.num_actions), DQN(args.state_dim, args.num_actions)
        self.lr = args.lr

    def load_model(self, file_name):
        checkpoint = torch.load((os.path.join(self.args.model_path, "{}_model.pth".format(file_name))))
        self.q_network.load_state_dict(checkpoint['q_state_dict'])
        self.optimizer_DQN.load_state_dict(checkpoint['optim'])
        # self.q_network.eval()

    def save_model(self, root_path):
        saving_dict = {'q_state_dict': self.q_network.state_dict(),
                       'optim': self.optimizer_DQN.state_dict(),
                       'is_best': 1}
        torch.save(saving_dict, root_path)

    def build_memory(self):
        memory_list = []
        for i in range(self.num_phases):
            memory_list.append([ReplayMemory(self.memory_size) for j in range(self.num_actions)])
        return memory_list

    def choose(self, count, state):

        ''' choose the best action for current state '''

        q_values = self.q_network(state)
        # print(q_values)
        if random.random() <= self.EPSILON:  # continue explore new Random Action
            self.action = random.randrange(len(q_values[0]))
            print("##Explore")
        else:  # exploitation
            self.action = torch.argmax(q_values[0])
        if self.EPSILON > 0.001 and count >= 20000:
            self.EPSILON = self.EPSILON * 0.9999
        return self.action, q_values

    def remember(self, state, action, reward, next_state):

        """ log the history """
        self.memory[state[3]][action].push([state, action, reward, next_state])

    def forget(self):

        """ remove the old history if the memory is too large """

        for phase_i in range(self.num_phases):
            for action_i in range(self.num_actions):
                if len(self.memory[phase_i][action_i]) > self.memory_size:
                    print("length of memory (state {0}, action {1}): {2}, before forget".format(
                        phase_i, action_i, len(self.memory[phase_i][action_i])))
                    self.memory[phase_i][action_i] = self.memory[phase_i][action_i][-self.memory_size:]
                print("length of memory (state {0}, action {1}): {2}, after forget".format(
                    phase_i, action_i, len(self.memory[phase_i][action_i])))

    def define_criterion_and_opti(self, device, weight_decay=1e-5):
        self.optimizer_DQN = torch.optim.Adam(params=self.q_network.parameters(),
                                              lr=self.lr,
                                              weight_decay=weight_decay)
        # self.scheduler_DQN = torch.optim.lr_scheduler.MultiStepLR(self.optim_DQN,
        #                                                           milestones=self.opt.schdr_step_size,
        #                                                           gamma=self.opt.schdr_gamma)

    def reset_update_count(self):

        self.q_target_outdated = 0

    def train_net(self, transitions):
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.q_network(state_batch).gather(1, action_batch)
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.q_target(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss().to(self.device)
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer_DQN.zero_grad()
        loss.backward()
        for param in self.q_network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer_DQN.step()

    def trainer(self):
        if self.batch_size > self.memory_size:
            return

        transitions = deque()

        for phase_i in range(self.num_phases):
            for action_i in range(self.num_actions):
                transitions += self.memory[phase_i][action_i].memory
        # add
        self.define_criterion_and_opti(self.device)
        self.train_net(transitions)
        self.q_target_outdated += 1
        self.forget()

    def update_network_bar(self):

        ''' update Q bar '''

        if self.q_target_outdated >= self.UPDATE_Q_TAR:
            self.q_target.load_state_dict(self.q_network.state_dict())
            self.q_target_outdated = 0

    @staticmethod
    def _cal_priority(sample_weight):
        pos_constant = 0.0001
        alpha = 1
        sample_weight = torch.Tensor(sample_weight)
        sample_weight_pr = torch.pow(sample_weight + pos_constant, alpha) / sample_weight.sum()
        return sample_weight_pr
