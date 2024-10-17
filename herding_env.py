import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Union
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from IPython.display import clear_output



#判断矩阵的邻居
def get_neighbors(matrix, row, col):
    rows, cols = matrix.shape
    
    # 定义可能的偏移量，上下左右
    offsets = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
    
    # 计算所有邻居的位置
    neighbor_positions = offsets + np.array([row, col])
    
    # 过滤掉越界的邻居
    valid_neighbors = (neighbor_positions[:, 0] >= 0) & (neighbor_positions[:, 0] < rows) & \
                      (neighbor_positions[:, 1] >= 0) & (neighbor_positions[:, 1] < cols)
    
    # 返回有效的邻居坐标
    return neighbor_positions[valid_neighbors].tolist()





def random_matrix_with_sum_1(size):
    # 生成随机正数矩阵
    matrix = np.random.rand(size, size)  # 生成介于 [0, 1) 的随机数矩阵
    
    # 归一化，使矩阵中所有元素的总和为 1
    matrix /= np.sum(matrix)
    
    # 保留三位小数
    matrix = np.round(matrix, 3)
    
    # 为了保证总和严格为 1，需要调整最后一个元素
    difference = 1.0 - np.sum(matrix)
    
    # 将差值加到矩阵中某个元素上，通常选最后一个元素
    matrix[-1, -1] += difference
    
    return matrix


def normalize_and_find_max_index(vector):
    # 将向量归一化，使其成为概率分布
    probability_vector = vector / np.sum(vector)
    
    # 找到概率最大的值的索引
    max_index = np.argmax(probability_vector)
    
    return max_index

def get_neighbors_gpu(matrix, row, col):
    # 将输入的矩阵转换为 GPU 上的张量
    matrix = matrix.to('cuda')
    rows, cols = matrix.shape

    # 定义邻居的相对位置：上、下、左、右
    neighbor_offsets = torch.tensor([[-1, 0], [1, 0], [0, -1], [0, 1]], device='cuda')

    # 计算邻居的绝对位置
    current_position = torch.tensor([row, col], device='cuda')
    neighbors = neighbor_offsets + current_position

    # 筛选合法的邻居（确保行列索引不越界）
    valid_mask = (neighbors[:, 0] >= 0) & (neighbors[:, 0] < rows) & \
                 (neighbors[:, 1] >= 0) & (neighbors[:, 1] < cols)
    valid_neighbors = neighbors[valid_mask]

    # 返回 GPU 上的结果，也可以调用 .cpu() 转为 CPU 张量
    return valid_neighbors


def random_matrix_with_sum_1_gpu(size):
    # 在 GPU 上生成随机正数矩阵
    matrix = torch.rand(size[0], size[1])  # 生成介于 [0, 1) 的随机数矩阵

    # 归一化，使矩阵中所有元素的总和为 1
    matrix /= torch.sum(matrix)

    # 保留三位小数
    matrix = torch.round(matrix * 1000) / 1000

    # 为了保证总和严格为 1，需要调整最后一个元素
    difference = 1.0 - torch.sum(matrix)

    # 将差值加到矩阵中的最后一个元素
    matrix[-1, -1] += difference

    return matrix

def normalize_and_find_max_index_gpu(vector):
    # 将向量移到 GPU 上
    vector = vector.to('cuda')

    # 将向量归一化，使其成为概率分布
    probability_vector = vector / torch.sum(vector)

    # 找到概率最大的值的索引
    max_index = torch.argmax(probability_vector)

    return max_index.item()  # 返回 CPU 上的索引值


class HerdingEnv(gym.Env):
    def __init__(self, size = 3, population = 1000, num_envs = 4096, device=torch.device("cuda:0"), gpu = False):
        super(HerdingEnv, self).__init__()
        self.size = size
        self.population = population
        self.num_envs = num_envs
        self.device = device
        self.gpu = gpu
        # 定义动作空间和观测空间
        self.action_space = spaces.Discrete(5)  # 上下左右stay五个动作
        self.observation_space = spaces.Box(low=0, high=1, shape=(2,2), dtype=np.int8)

        # 初始化环境状态
        self.jump_rate = 0.05
        
        self.follower_state = None
        self.target_state = None
        self.max_steps = 1000  # 定义最大步数
        self.current_steps = 0
        self.difference = None

    def reset_env_i(self, i):
        self.follower_state_batch[i] = random_matrix_with_sum_1_gpu(self.size)
        self.target_state_batch[i] = random_matrix_with_sum_1_gpu(self.size)
        self.leader_state_batch[i] = torch.randint(0, 2, (self.num_envs, 2), device=self.device)

        self.current_steps[i] = 0      

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None,):
        # 重置环境状态并返回初始观测值和信息字典
        super().reset(seed=seed)
        if self.gpu:
            self.follower_state_batch = torch.zeros((self.num_envs, self.size**2),\
                                                    dtype=torch.float32, device=self.device)
            self.target_state_batch = torch.zeros((self.num_envs, self.size**2),\
                                                    dtype=torch.float32, device=self.device)
            self.leader_state_batch = torch.zeros((self.num_envs, 2), device=self.device)
            self.current_steps = torch.zeros(self.num_envs, dtype=torch.int8, device=self.device)
        else:
            self.follower_state = random_matrix_with_sum_1(self.size)
            self.target_state = random_matrix_with_sum_1(self.size)
            self.leader_state = np.random.randint(0, 2, size=2)
            self.current_steps = 0
        if self.gpu:
            for env_i in range(self.num_envs):
                self.reset_env_i(env_i)
            
        info = {}
        if self.gpu:
            self.state = torch.cat((self.follower_state_batch, self.target_state_batch, self.leader_state_batch))
        else:
            self.state = np.concatenate((self.leader_state, self.target_state.flatten(), self.follower_state.flatten()))
        return self.state, info

    def step(self, action):
        if self.gpu:
            pass
        else:
        # 执行动作并返回新的观测值、奖励、完成标志、截断标志和信息字典
            if action.size > 1:
                action = normalize_and_find_max_index(action)
            assert self.action_space.contains(action), f"Invalid Action: {action}"

            # 领导者状态随动作变化
            action_moves = np.array([
            [-1, 0],  # action 0: 上移
            [1, 0],   # action 1: 下移
            [0, -1],  # action 2: 左移
            [0, 1]    # action 3: 右移
            ])

            # 更新 leader_state
            if action in [0, 1, 2, 3]:
                self.leader_state += action_moves[action]
                # 使用 clip 限制 leader_state 的范围
                self.leader_state = np.clip(self.leader_state, 0, self.size - 1)

            # 计算 follower_state 和 target_state 对应元素的最小值的总和
            last_cover = np.minimum(self.follower_state, self.target_state).sum()

            #跟随者状态随领导者变化
            if action == 4:
                # 获取领导者当前的坐标
                leader_x, leader_y = self.leader_state

                # 获取领导者位置的邻居
                neighbors = get_neighbors(self.follower_state, leader_x, leader_y)
                num_followers = max(0, int(self.follower_state[leader_x, leader_y] * self.population))

                # 随机生成跳跃的布尔掩码
                jump_mask = np.random.rand(num_followers) < self.jump_rate

                # 统计需要跳跃的个体数
                num_jumps = np.sum(jump_mask)

                if num_jumps > 0:
                    # 随机选择邻居进行跳跃
                    chosen_neighbors = np.random.choice(len(neighbors), size=num_jumps)

                    # 减少领导者位置的个体数
                    self.follower_state[leader_x, leader_y] -= num_jumps / self.population

                    # 增加对应邻居位置的个体数
                    for idx in chosen_neighbors:
                        neighbor_x, neighbor_y = neighbors[idx][0], neighbors[idx][1]
                        self.follower_state[neighbor_x, neighbor_y] += 1 / self.population

            # 定义奖励函数
            cover = 0
            cover = np.minimum(self.follower_state, self.target_state).sum()
            reward = (cover - last_cover) 
            # #比较一下
            # reward = -np.sum(np.square(self.follower_state - self.target_state))
            
            
            # 更新步数
            self.current_steps += 1

            # 判断是否完成
            done = self.current_steps >= self.max_steps
            truncated = False  # 如果有时间限制，可以设置为 True
            # self.state = np.concatenate((self.leader_state, self.follower_state.flatten()/1000))
            self.state = np.concatenate((self.leader_state, self.target_state.flatten(), self.follower_state.flatten()))
            info = {}
            truncated = False
            self.difference = np.sum(np.abs(self.follower_state - self.target_state))
            if  self.difference < 20/self.population:
                truncated = True
            return self.state, reward, done, truncated, info


        

    def render(self):
        clear_output(wait=True)
        plt.imshow(self.follower_state/self.population, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Number of Robots')
        plt.title('Robot Distribution in Grid')
        plt.show()
        print(f"difference: {self.difference}")

    

    def close(self):
        
        pass

