# src/algorithms/drl_tsbc.py
"""
DRL-TSBC: Deep Reinforcement Learning-based dynamic bus Timetable Scheduling method with Bidirectional Constraints
基于深度强化学习的双向动态公交时刻表排班算法实现

基于论文: 基于交通大数据的公交排班和调度机制研究 - 谢嘉昊, 王玺钧
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from typing import Tuple, List, Dict, Any

class DQNNetwork(nn.Module):
    """DQN神经网络 - 用于DRL-TSBC双向时刻表排班"""
    
    def __init__(self, state_dim: int = 10, hidden_dims: List[int] = [128, 256, 128], action_dim: int = 4):
        """
        初始化DQN网络
        
        Args:
            state_dim: 状态维度 (双向状态特征)
            hidden_dims: 隐藏层维度列表
            action_dim: 动作维度 (4种组合: 00, 01, 10, 11)
        """
        super(DQNNetwork, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        # 构建隐藏层
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU()
            ])
            input_dim = hidden_dim
            
        # 输出层
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        
        return (
            torch.FloatTensor(state),
            torch.LongTensor(action),
            torch.FloatTensor(reward),
            torch.FloatTensor(next_state),
            torch.BoolTensor(done)
        )
        
    def __len__(self) -> int:
        return len(self.buffer)

class DRLTSBCAgent:
    """DRL-TSBC智能体 - 双向动态公交时刻表排班"""
    
    def __init__(self, 
                 state_dim: int = 10,  # 双向状态特征
                 action_dim: int = 4,   # 4种动作组合
                 learning_rate: float = 0.001,
                 gamma: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 batch_size: int = 32,
                 target_update_freq: int = 1000):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # 神经网络
        self.q_network = DQNNetwork(state_dim, action_dim=action_dim)
        self.target_network = DQNNetwork(state_dim, action_dim=action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # 经验回放
        self.replay_buffer = ReplayBuffer()
        
        # 训练计数器
        self.train_step = 0
        
        # 约束参数
        self.t_min = 2   # 最小发车间隔(分钟)
        self.t_max = 15  # 最大发车间隔(分钟)
        
        # 动作映射: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        self.action_map = {
            0: (0, 0),  # 上下行都不发车
            1: (0, 1),  # 上行不发车，下行发车
            2: (1, 0),  # 上行发车，下行不发车
            3: (1, 1)   # 上下行都发车
        }
        
        # 同步目标网络
        self.update_target_network()
        
    def update_target_network(self):
        """更新目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def get_state_features(self, env_state: Dict[str, Any]) -> np.ndarray:
        """
        提取双向状态特征
        
        根据论文，状态包含双向的公交线路信息:
        - 时间特征
        - 上行方向特征 
        - 下行方向特征
        - 双向约束特征
        """
        features = np.array([
            # 时间特征
            env_state['time_hour'] / 24.0,
            env_state['time_minute'] / 60.0,
            
            # 上行方向特征
            env_state['up_capacity_utilization'],     # 上行运力利用率
            env_state['up_waiting_time'],             # 上行等待时间
            env_state['up_stranded_passengers'],      # 上行滞留乘客
            
            # 下行方向特征  
            env_state['down_capacity_utilization'],   # 下行运力利用率
            env_state['down_waiting_time'],           # 下行等待时间
            env_state['down_stranded_passengers'],    # 下行滞留乘客
            
            # 双向约束特征
            env_state['departure_count_diff'],        # 上下行发车次数差
            env_state['total_buses_in_service']       # 总在途车辆数
        ])
        return features
        
    def select_action(self, state: np.ndarray, last_intervals: Dict[str, int]) -> Tuple[int, int]:
        """
        选择动作（带双向约束的ε-贪婪策略）
        
        Args:
            state: 当前状态特征
            last_intervals: {'up': 上行间隔, 'down': 下行间隔}
            
        Returns:
            (a_up, a_down): 上行和下行的发车决策
        """
        
        # 先进行约束检查
        up_forced = None
        down_forced = None
        
        # 上行约束检查
        if last_intervals['up'] < self.t_min:
            up_forced = 0  # 强制不发车
        elif last_intervals['up'] >= self.t_max:
            up_forced = 1  # 强制发车
            
        # 下行约束检查  
        if last_intervals['down'] < self.t_min:
            down_forced = 0  # 强制不发车
        elif last_intervals['down'] >= self.t_max:
            down_forced = 1  # 强制发车
            
        # 如果都有强制约束，直接返回
        if up_forced is not None and down_forced is not None:
            return (up_forced, down_forced)
            
        # ε-贪婪策略选择动作
        if random.random() < self.epsilon:
            # 随机选择（考虑约束）
            valid_actions = []
            for action_idx, (a_up, a_down) in self.action_map.items():
                if (up_forced is None or a_up == up_forced) and \
                   (down_forced is None or a_down == down_forced):
                    valid_actions.append(action_idx)
            
            if valid_actions:
                action_idx = random.choice(valid_actions)
            else:
                action_idx = 0  # 默认都不发车
        else:
            # 贪婪选择
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_network(state_tensor).squeeze()
                
                # 应用约束掩码
                masked_q_values = q_values.clone()
                for action_idx, (a_up, a_down) in self.action_map.items():
                    if (up_forced is not None and a_up != up_forced) or \
                       (down_forced is not None and a_down != down_forced):
                        masked_q_values[action_idx] = float('-inf')
                        
                action_idx = masked_q_values.argmax().item()
                
        return self.action_map[action_idx]
                
    def calculate_reward(self, action: Tuple[int, int], env_state: Dict[str, Any]) -> float:
        """
        计算双向约束奖励函数
        
        根据论文公式(2.16)和(2.17)计算上下行奖励
        """
        a_up, a_down = action
        
        # 上行奖励计算
        o_up = env_state['up_capacity_utilization']
        w_up = env_state['up_waiting_time'] 
        d_up = env_state['up_stranded_passengers']
        c_up = env_state['up_departure_count']
        c_down = env_state['down_departure_count']
        
        # 下行奖励计算
        o_down = env_state['down_capacity_utilization']
        w_down = env_state['down_waiting_time']
        d_down = env_state['down_stranded_passengers']
        
        # 奖励函数参数
        omega = 1.0 / 4000.0  # 等待时间惩罚因子
        beta = 0.2            # 滞留乘客惩罚因子  
        zeta = 0.1            # 双向约束因子
        
        # 上行奖励 (公式2.16)
        if a_up == 0:  # 不发车
            r_up = (1 - o_up) - (omega * w_up) - (beta * d_up) - zeta * (c_up - c_down)
        else:  # 发车
            r_up = o_up - (beta * d_up) + zeta * (c_up - c_down)
            
        # 下行奖励 (公式2.17)  
        if a_down == 0:  # 不发车
            r_down = (1 - o_down) - (omega * w_down) - (beta * d_down) + zeta * (c_up - c_down)
        else:  # 发车
            r_down = o_down - (beta * d_down) - zeta * (c_up - c_down)
            
        # 总奖励
        total_reward = r_up + r_down
        
        return total_reward
        
    def train(self):
        """训练智能体"""
        if len(self.replay_buffer) < self.batch_size:
            return None
            
        # 采样批次数据
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # 计算当前Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
            
        # 计算损失
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新ε
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        # 更新目标网络
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.update_target_network()
            
        return loss.item()
        
    def action_to_index(self, action: Tuple[int, int]) -> int:
        """将动作元组转换为索引"""
        for idx, mapped_action in self.action_map.items():
            if mapped_action == action:
                return idx
        return 0  # 默认返回0
        
    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_step': self.train_step
        }, filepath)
        
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.train_step = checkpoint['train_step']

class BidirectionalTimetableOptimizer:
    """双向时刻表优化器"""
    
    def __init__(self, agent: DRLTSBCAgent):
        self.agent = agent
        
    def optimize_timetable(self, env, episodes: int = 1000) -> List[Dict[str, Any]]:
        """
        优化双向时刻表
        
        Args:
            env: 双向公交仿真环境
            episodes: 训练轮数
            
        Returns:
            训练历史记录
        """
        training_history = []
        
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            episode_actions = []
            last_intervals = {'up': 0, 'down': 0}
            
            while not env.is_done():
                # 获取状态特征
                state_features = self.agent.get_state_features(state)
                
                # 选择动作
                action = self.agent.select_action(state_features, last_intervals)
                
                # 执行动作
                next_state, _, done = env.step(action)
                
                # 计算实际奖励
                actual_reward = self.agent.calculate_reward(action, state)
                
                # 存储经验
                next_state_features = self.agent.get_state_features(next_state)
                action_idx = self.agent.action_to_index(action)
                
                self.agent.replay_buffer.push(
                    state_features, action_idx, actual_reward, next_state_features, done
                )
                
                # 训练
                loss = self.agent.train()
                
                # 更新状态和间隔
                state = next_state
                episode_reward += actual_reward
                episode_actions.append(action)
                
                # 更新发车间隔
                if action[0] == 1:  # 上行发车
                    last_intervals['up'] = 0
                else:
                    last_intervals['up'] += 1
                    
                if action[1] == 1:  # 下行发车
                    last_intervals['down'] = 0
                else:
                    last_intervals['down'] += 1
                    
            # 记录训练历史
            training_history.append({
                'episode': episode,
                'reward': episode_reward,
                'epsilon': self.agent.epsilon,
                'actions': episode_actions,
                'loss': loss if loss else 0,
                'up_departures': sum(1 for a in episode_actions if a[0] == 1),
                'down_departures': sum(1 for a in episode_actions if a[1] == 1)
            })
            
            if episode % 100 == 0:
                up_deps = training_history[-1]['up_departures']
                down_deps = training_history[-1]['down_departures']
                print(f"Episode {episode}, Reward: {episode_reward:.2f}, "
                      f"Up/Down Departures: {up_deps}/{down_deps}, "
                      f"Epsilon: {self.agent.epsilon:.3f}")
                
        return training_history