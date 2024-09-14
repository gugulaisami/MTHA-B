import gymnasium as gym
env = gym.make(
    "LunarLander-v2", render_mode="human",
     continuous= False
    # gravity = -10.0,
    # enable_wind= = False,
    # wind_power= 15.0,
    # turbulence_power = 1.5
)

import gymnasium as gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from IPython.display import HTML, display
import imageio
import base64
import io
import glob
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from collections import deque, namedtuple
import keyboard #使其能够监听键盘事件
import sys

######## 创建网络架构
class Network(nn.Module):
    def __init__(self, state_size, action_size, seed=42):
        super(Network, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.dropout1 = nn.Dropout(p=0.3)  # p=0.5 是 dropout rate
        self.fc2 = nn.Linear(64, 64)
        self.dropout2 = nn.Dropout(p=0.1)  # 另一个dropout 层
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.dropout1(x)  # 在 forward 方法中使用 dropout 层
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)  # 另一个 dropout 层
        return self.fc3(x)

######################################################################################################
def get_human_action():
    key = keyboard.read_key()
    action_map = {'1': 0, '2': 1, '3': 2, '4': 3}
    return action_map.get(key, -1)

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def get_monte_carlo_predictions(data_loader,
                                forward_passes,
                                model,
                                n_classes,
                                n_samples):
    """ Function to get the monte-carlo samples and uncertainty estimates
    through multiple forward passes

    Parameters
    ----------
    data_loader : object
        data loader object from the data loader module
    forward_passes : int
        number of monte-carlo samples/forward passes
    model : object
        keras model
    n_classes : int
        number of classes in the dataset
    n_samples : int
        number of samples in the test set
    """

    dropout_predictions = np.empty((0, n_samples, n_classes))
    softmax = nn.Softmax(dim=1)
    for i in range(forward_passes):
        predictions = np.empty((0, n_classes))
        model.eval()
        enable_dropout(model)#使得模型的dropout层参与预测
        for i, (image, label) in enumerate(data_loader):#其实label用不到，关键只需要state，不需要action
            image = image.to(torch.device('cuda'))
            with torch.no_grad():
                output = model(image)
                output = softmax(output)  # shape (n_samples, n_classes)
            predictions = np.vstack((predictions, output.cpu().numpy()))

        dropout_predictions = np.vstack((dropout_predictions,
                                         predictions[np.newaxis, :, :]))
        # dropout predictions - shape (forward_passes, n_samples, n_classes)

    # Calculating mean across multiple MCD forward passes
    mean = np.mean(dropout_predictions, axis=0)  # shape (n_samples, n_classes)

    # Calculating variance across multiple MCD forward passes
    variance = np.var(dropout_predictions, axis=0)  # shape (n_samples, n_classes)

    epsilon = sys.float_info.min
    # Calculating entropy across multiple MCD forward passes
    entropy = -np.sum(mean * np.log(mean + epsilon), axis=-1)  # shape (n_samples,)

    # Calculating mutual information across multiple MCD forward passes
    mutual_info = entropy - np.mean(np.sum(-dropout_predictions * np.log(dropout_predictions + epsilon),
                                           axis=-1), axis=0)  # shape (n_samples,)
    #return variance#输出方差

#仲裁函数，根据机器和人的行为的可信度和优化目标函数值决定最终行为
def arbitration(machine_credibility, human_credibility,score_machine,score_human,score_boundary,
                machine_action, human_action,boundary_action):
    if machine_credibility >= human_credibility and score_machine>=score_human and score_machine>=score_boundary:
        print(f'choose machine_action: {machine_action}')
        return machine_action
    elif score_boundary>=score_human:
        print(f'choose boundary_action: {boundary_action}')
        return boundary_action
    else:
        print(f'choose human_action: {human_action}')
        return human_action
########################################################################################################

######## 设置环境：使用Gymnasium创建了LunarLander环境
state_shape = env.observation_space.shape
state_size = env.observation_space.shape[0]
number_actions = env.action_space.n
print('State shape: ', state_shape)
print('State size: ', state_size)
print('Number of actions: ', number_actions)
######## 初始化超参数：定义了学习率、批处理大小、折扣因子等超参数。
learning_rate = 5e-4
minibatch_size = 100
discount_factor = 0.99
replay_buffer_size = int(1e5)
interpolation_parameter = 1e-3


####### 实现经验回放：实现了经验回放（Experience Replay）的类 ReplayMemory，用于存储和采样Agent的经验
class ReplayMemory(object):
    def __init__(self, capacity):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.capacity = capacity
        self.memory = []

    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        experiences = random.sample(self.memory, k=batch_size)
        states = torch.from_numpy(np.vstack(
            [e[0] for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(
            np.vstack([e[1] for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack(
            [e[2] for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack(
            [e[3] for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(
            np.uint8)).float().to(self.device)
        return states, next_states, actions, rewards, dones

    def get_dataset(self):
        states = torch.from_numpy(np.vstack(
            [e[0] for e in self.memory if e is not None])).float().to(self.device)
        actions = torch.from_numpy(
            np.vstack([e[1] for e in self.memory if e is not None])).long().to(self.device)
        # rewards = torch.from_numpy(np.vstack(
        #     [e[2] for e in self.memory if e is not None])).float().to(self.device)
        # next_states = torch.from_numpy(np.vstack(
        #     [e[3] for e in self.memory if e is not None])).float().to(self.device)
        # dones = torch.from_numpy(np.vstack(
        #     [e[4] for e in self.memory if e is not None]).astype(np.uint8)).float().to(self.device)
        return states, actions


########## 实现 DQN 代理：创建了一个Agent类，包含本地Q网络和目标Q网络，包含了采取动作、学习、软更新等方法。
class Agent():
    # 初始化函数，参数为状态大小和动作大小
    def __init__(self, state_size, action_size):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.local_qnetwork = Network(state_size, action_size).to(self.device)
        self.target_qnetwork = Network(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(
            self.local_qnetwork.parameters(), lr=learning_rate)
        self.memory = ReplayMemory(replay_buffer_size)
        self.t_step = 0

    # 定义一个函数，用于存储经验并决定何时从中学习
    def step(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))
        self.t_step = (self.t_step + 1) % 4
        if self.t_step == 0:
            if len(self.memory.memory) > minibatch_size:
                experiences = self.memory.sample(100)
                self.learn(experiences, discount_factor)

    # 定义一个函数，根据给定的状态和epsilon值选择一个动作（epsilon贪婪动作选择策略）0.表示浮点数
    #def act(self, state, epsilon=0.):
    def act(self, state, epsilon=0., human_control=False):
        if human_control:
            return get_human_action()
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.local_qnetwork.eval()
        with torch.no_grad():
            action_values = self.local_qnetwork(state)#使用本地Q网络的输出
        self.local_qnetwork.train()
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    # 定义一个函数，根据样本经验更新代理的q值，参数为经验和折扣因子
    def learn(self, experiences, discount_factor):
        states, next_states, actions, rewards, dones = experiences
        next_q_targets = self.target_qnetwork(
            next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + discount_factor * next_q_targets * (1 - dones)
        q_expected = self.local_qnetwork(states).gather(1, actions)
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.local_qnetwork,
                         self.target_qnetwork, interpolation_parameter)

    # 定义一个函数，用于软更新目标网络的参数，参数为本地模型，目标模型和插值参数
    def soft_update(self, local_model, target_model, interpolation_parameter):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(interpolation_parameter * local_param.data + (
                    1.0 - interpolation_parameter) * target_param.data)

####### 创建 Agent 类的实例，传入状态维度和动作个数。
agent = Agent(state_size, number_actions)
# 加载训练好的模型权重
agent.local_qnetwork.load_state_dict(torch.load('checkpoint.pth'))
## 设置成预测模式
#agent.local_qnetwork.eval()

##### 训练DQN代理
number_episodes = 2000
maximum_number_timesteps_per_episode = 1000
#epsilon_starting_value = 1#随机选动作，exploration
epsilon_starting_value = 0.1#随机选动作，exploration
epsilon_ending_value = 0.01
epsilon_decay_value = 0.995
epsilon = epsilon_starting_value
scores_on_100_episodes = deque(maxlen=100)

# #预训练DQN
# for episode in range(1, number_episodes + 1):
#     state, _ = env.reset()
#     score = 0
#     for t in range(maximum_number_timesteps_per_episode):
#         action = agent.act(state, epsilon,human_control=False)#获取本地Q网络输出/键盘行为
#         if action == -1:
#             continue
#         next_state, reward, done, _, _ = env.step(action)
#         agent.step(state, action, reward, next_state, done)
#         state = next_state
#         score += reward
#         if done:
#             break
#     scores_on_100_episodes.append(score)
#     epsilon = max(epsilon_ending_value, epsilon_decay_value * epsilon)
#     print('\rEpisode {}\tAverage Score: {:.2f}'.format(
#         episode, np.mean(scores_on_100_episodes)), end="")
#     if episode % 100 == 0:
#         print('\rEpisode {}\tAverage Score: {:.2f}'.format(
#             episode, np.mean(scores_on_100_episodes)))
#     if np.mean(scores_on_100_episodes) >= 200.0:
#         print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
#             episode - 100, np.mean(scores_on_100_episodes)))
#         torch.save(agent.local_qnetwork.state_dict(), 'checkpoint.pth')
#         break

# #创建新的预测环境
# env2 = gym.make(
#     "LunarLander-v2", render_mode="human",
#      continuous= False
#     # gravity = -10.0,
#     # enable_wind= = False,
#     # wind_power= 15.0,
#     # turbulence_power = 1.5
# )

# # 自定义数据集类
# from torch.utils.data import Dataset, DataLoader
# class LunarLanderDataset(Dataset):
#     def __init__(self, env, agent, n_samples):
#         self.env = env
#         self.agent = agent
#         self.n_samples = n_samples
#         self.data = self.collect_data()
#
#     def collect_data(self):
#         data = []
#         for _ in range(self.n_samples):
#             state, _ = self.env.reset()
#             done = False
#             while not done:
#                 action = self.agent.act(state)
#                 next_state, reward, done, _, _ = self.env.step(action)
#                 data.append((state, action))
#                 state = next_state
#         return data
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         state, action = self.data[idx]
#         state = torch.from_numpy(state).float()
#         action = torch.tensor(action).long()
#         return state, action

#设置mc dropout参数
model = agent.local_qnetwork  # 训练好的模型
forward_passes = 10  # Monte Carlo采样次数/前向传播次数，要设置成500
n_samples = 100  # 样本数量
n_features = state_size  # 特征数量（状态数量）
n_classes = number_actions  # 类别数量（行动数量）

#决策函数
for episode in range(1, number_episodes + 1):
    state, _ = env.reset()
    score = 0
    score_machine=0
    score_human=0
    boundary_human=0
    reward_max=0
    score_boundary=0
    for t in range(maximum_number_timesteps_per_episode):
        dropout_predictions = []
        human_predictions = []

        #评估机器/人的信度
        for i in range(forward_passes):
            #predictions = np.empty((0, 1))
            agent.local_qnetwork.eval()#使得本地Q网络进入预测模式
            enable_dropout(agent.local_qnetwork)#使得模型的dropout层参与预测
            action_machine = agent.act(state, epsilon, human_control= False)#获取本地网络输出/键盘行为
            action_human = agent.act(state, epsilon, human_control=True)#获取本地网络输出/键盘行为
            dropout_predictions.append(action_machine)#存储包含mc dropout的DQN行动输出
            human_predictions.append(action_human)#存储键盘输入

        # #经过多次前向传播，计算均值
        # mean = np.mean(np.array(dropout_predictions))
        # 经过多次前向传播，计算方差
        variance_machine = np.var(np.array(dropout_predictions))
        variance_human = np.var(np.array(human_predictions))
        credibility_machine = 1-variance_machine  # 方差越大，可信度越小
        credibility_human=1-variance_human  #方差越大，可信度越小
        print(f'Episode {episode}, Step {t}: credibility_machine: {credibility_machine}')
        print(f'Episode {episode}, Step {t}: credibility_human: {credibility_human}')

        _,reward_machine,_,_,_=env.step(agent.act(state, epsilon,human_control=False))
        _,reward_human, _, _, _ = env.step(agent.act(state, epsilon, human_control=True))
        _,reward_boundary, _, _, _ = env.step(boundary_human)#使用t-1时刻的boundary
        score_boundary+=reward_boundary
        score_machine+=reward_machine
        score_human += reward_human
        print(f'Episode {episode}, Step {t}: score_machine: {score_machine}')
        print(f'Episode {episode}, Step {t}: score_human: {score_human}')
        print(f'Episode {episode}, Step {t}: score_boundary: {score_boundary}')
        #等待决策完之后再更新boundary
        if reward_human>reward_max :
            reward_max=reward_human
            boundary_human=agent.act(state, epsilon, human_control=True)

        #仲裁函数
        action = arbitration(credibility_machine, credibility_human, score_machine,score_human,score_boundary,agent.act(state, epsilon, human_control=False),
                             agent.act(state, epsilon, human_control=True),boundary_human)
        if action == -1:
            continue
        next_state, reward, done, _, _ = env.step(action)
        score += reward
        agent.step(state, action, reward, next_state, done)#从经验中进行学习
        state = next_state#切换状态

        # print(f'Episode {episode}, Step {t}: score_machine: {score_machine}')
        # print(f'Episode {episode}, Step {t}: score_human: {score_human}')

        if done:
            break
    scores_on_100_episodes.append(score)
    epsilon = max(epsilon_ending_value, epsilon_decay_value * epsilon)
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(
        episode, np.mean(scores_on_100_episodes)), end="")
    if episode % 100 == 0:
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(
            episode, np.mean(scores_on_100_episodes)))


####### 可视化结果
def show_video_of_model(agent, env_name):
    env = gym.make(env_name, render_mode='rgb_array')
    state, _ = env.reset()
    done = False
    frames = []
    while not done:
        frame = env.render()
        frames.append(frame)
        action = agent.act(state)
        state, reward, done, _, _ = env.step(action.item())
    env.close()
    imageio.mimsave('video.mp4', frames, fps=30)


show_video_of_model(agent, 'LunarLander-v2')


def show_video():
    mp4list = glob.glob('*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        display(HTML(data='''<video alt="test" autoplay
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
    else:
        print("Could not find video")


show_video()