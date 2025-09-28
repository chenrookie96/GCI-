"""
DRL-TSBC: Deep Reinforcement Learning-based dynamic bus Timetable Scheduling 
with Bidirectional Constraints

基于深度强化学习的双向动态公交时刻表排班算法
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Tuple, List, Dict, Any

class DRLTSBCAgent:
    """DRL-TSBC算法智能体"""
    
    def __init__(self, state_dim: int = 10, action_dim: int = 4, 
                 learning_rate: float = 0.001, gamma: float = 0.95,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01, memory_size: int = 10000,
                 batch_size: int = 32):
        """
        初始化DRL-TSBC智能体
        
        Args:
            state_dim: 状态空间维度 (10维：时间2维 + 上行4维 + 下行4维)
            action_dim: 动作空间维度 (4种组合)
            learning_rate: 学习率
            gamma: 折扣因子
            epsilon: 探索率
            epsilon_decay: 探索率衰减
            epsilon_min: 最小探索率
            memory_size: 经验回放缓冲区大小
            batch_size: 批次大小
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        
        # 神经网络
        self.q_network = self._build_network()
        self.target_network = self._build_network()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # 经验回放
        self.memory = deque(maxlen=memory_size)
        
        # 更新目标网络
        self.update_target_network()
        
    def _build_network(self) -> nn.Module:
        """构建DQN网络"""
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim)
        )
    
    def update_target_network(self):
        """更新目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def get_state(self, env_state: Dict[str, Any]) -> np.ndarray:
        """
        构建状态向量
        
        Args:
            env_state: 环境状态信息
            
        Returns:
            状态向量 [a1_m, a2_m, x1_m, x2_m, x3_m, x4_m, y1_m, y2_m, y3_m, y4_m]
        """
        # 时间状态
        current_hour = env_state['current_hour']
        current_minute = env_state['current_minute']
        a1_m = current_hour / 24.0
        a2_m = current_minute / 60.0
        
        # 上行方向状态
        x1_m = env_state['up_max_load_rate']  # 满载率
        x2_m = env_state['up_waiting_time'] / env_state['mu']  # 等待时间
        x3_m = env_state['up_capacity_utilization']  # 容量利用率
        x4_m = env_state['up_departure_count'] / env_state['delta']  # 发车次数
        
        # 下行方向状态
        y1_m = env_state['down_max_load_rate']
        y2_m = env_state['down_waiting_time'] / env_state['mu']
        y3_m = env_state['down_capacity_utilization']
        y4_m = env_state['down_departure_count'] / env_state['delta']
        
        return np.array([a1_m, a2_m, x1_m, x2_m, x3_m, x4_m, y1_m, y2_m, y3_m, y4_m])
    
    def choose_action(self, state: np.ndarray) -> int:
        """
        选择动作
        
        Args:
            state: 当前状态
            
        Returns:
            动作索引 (0-3对应四种发车组合)
        """
        if random.random() <= self.epsilon:
            return random.randrange(self.action_dim)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                 next_state: np.ndarray, done: bool):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """经验回放训练"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def calculate_reward(self, env_state: Dict[str, Any], action: int) -> float:
        """
        计算奖励函数
        
        Args:
            env_state: 环境状态
            action: 执行的动作
            
        Returns:
            奖励值
        """
        # 解析动作
        up_departure = action in [2, 3]  # 上行发车
        down_departure = action in [1, 3]  # 下行发车
        
        reward = 0.0
        
        # 上行方向奖励
        if up_departure:
            # 容量利用率奖励
            reward += env_state['up_capacity_utilization']
            # 滞留乘客惩罚
            if env_state['up_stranded_passengers'] > 0:
                reward -= 0.5 * env_state['up_stranded_passengers']
        else:
            # 不发车时的奖励
            reward += (1 - env_state['up_capacity_utilization'])
            # 等待时间惩罚
            reward -= 0.1 * env_state['up_waiting_time']
        
        # 下行方向奖励
        if down_departure:
            reward += env_state['down_capacity_utilization']
            if env_state['down_stranded_passengers'] > 0:
                reward -= 0.5 * env_state['down_stranded_passengers']
        else:
            reward += (1 - env_state['down_capacity_utilization'])
            reward -= 0.1 * env_state['down_waiting_time']
        
        # 发车次数一致性奖励
        departure_diff = abs(env_state['up_departure_count'] - env_state['down_departure_count'])
        reward -= 0.2 * departure_diff
        
        return reward

class DRLTSBCEnvironment:
    """DRL-TSBC仿真环境"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化环境
        
        Args:
            config: 环境配置参数
        """
        self.config = config
        self.current_time = 0
        self.up_departure_count = 0
        self.down_departure_count = 0
        self.passenger_flow_data = self._load_passenger_flow()
        
    def _load_passenger_flow(self) -> Dict[str, Any]:
        """加载客流数据"""
        # 这里应该加载真实的客流数据
        # 暂时返回模拟数据
        return {
            'up_flow': np.random.poisson(10, 1440),  # 上行客流
            'down_flow': np.random.poisson(10, 1440),  # 下行客流
        }
    
    def reset(self) -> Dict[str, Any]:
        """重置环境"""
        self.current_time = 0
        self.up_departure_count = 0
        self.down_departure_count = 0
        return self._get_state()
    
    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        执行一步动作
        
        Args:
            action: 动作 (0-3)
            
        Returns:
            (next_state, reward, done, info)
        """
        # 解析动作
        up_departure = action in [2, 3]
        down_departure = action in [1, 3]
        
        # 更新发车次数
        if up_departure:
            self.up_departure_count += 1
        if down_departure:
            self.down_departure_count += 1
        
        # 更新时间
        self.current_time += 1
        
        # 获取新状态
        next_state = self._get_state()
        
        # 计算奖励
        reward = self._calculate_reward(action)
        
        # 判断是否结束
        done = self.current_time >= 1440  # 24小时 = 1440分钟
        
        info = {
            'up_departures': self.up_departure_count,
            'down_departures': self.down_departure_count,
            'time': self.current_time
        }
        
        return next_state, reward, done, info
    
    def _get_state(self) -> Dict[str, Any]:
        """获取当前状态"""
        current_hour = self.current_time // 60
        current_minute = self.current_time % 60
        
        # 获取当前时刻的客流数据
        up_flow = self.passenger_flow_data['up_flow'][self.current_time]
        down_flow = self.passenger_flow_data['down_flow'][self.current_time]
        
        # 计算各种状态指标
        up_max_load_rate = min(up_flow / self.config['bus_capacity'], 1.0)
        down_max_load_rate = min(down_flow / self.config['bus_capacity'], 1.0)
        
        up_waiting_time = up_flow * 2.0  # 模拟等待时间
        down_waiting_time = down_flow * 2.0
        
        up_capacity_utilization = up_flow / (self.config['bus_capacity'] * 1.5)
        down_capacity_utilization = down_flow / (self.config['bus_capacity'] * 1.5)
        
        up_stranded_passengers = max(0, up_flow - self.config['bus_capacity'])
        down_stranded_passengers = max(0, down_flow - self.config['bus_capacity'])
        
        return {
            'current_hour': current_hour,
            'current_minute': current_minute,
            'up_max_load_rate': up_max_load_rate,
            'down_max_load_rate': down_max_load_rate,
            'up_waiting_time': up_waiting_time,
            'down_waiting_time': down_waiting_time,
            'up_capacity_utilization': up_capacity_utilization,
            'down_capacity_utilization': down_capacity_utilization,
            'up_departure_count': self.up_departure_count,
            'down_departure_count': self.down_departure_count,
            'up_stranded_passengers': up_stranded_passengers,
            'down_stranded_passengers': down_stranded_passengers,
            'mu': 100.0,  # 归一化参数
            'delta': 50.0  # 归一化参数
        }
    
    def _calculate_reward(self, action: int) -> float:
        """计算奖励"""
        # 这里实现具体的奖励计算逻辑
        # 简化版本
        reward = 0.0
        
        up_departure = action in [2, 3]
        down_departure = action in [1, 3]
        
        if up_departure:
            reward += 0.1
        if down_departure:
            reward += 0.1
        
        # 发车次数一致性奖励
        departure_diff = abs(self.up_departure_count - self.down_departure_count)
        reward -= 0.05 * departure_diff
        
        return reward

def train_drl_tsbc(episodes: int = 1000, save_path: str = "models/drl_tsbc.pth"):
    """训练DRL-TSBC模型"""
    
    # 环境配置
    config = {
        'bus_capacity': 50,
        'max_departure_interval': 30,
        'min_departure_interval': 5
    }
    
    # 创建环境和智能体
    env = DRLTSBCEnvironment(config)
    agent = DRLTSBCAgent()
    
    # 训练循环
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        while True:
            # 获取状态向量
            state_vector = agent.get_state(state)
            
            # 选择动作
            action = agent.choose_action(state_vector)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            next_state_vector = agent.get_state(next_state)
            
            # 存储经验
            agent.remember(state_vector, action, reward, next_state_vector, done)
            
            # 训练
            agent.replay()
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        # 更新目标网络
        if episode % 10 == 0:
            agent.update_target_network()
        
        # 打印训练进度
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    # 保存模型
    torch.save(agent.q_network.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_drl_tsbc()
