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
    """DQN神经网络 - 用于DRL-TSBC双向时刻表排班
    
    根据论文表2-2规范:
    - 12个隐藏层
    - 每层500个神经元
    - 激活函数: ReLU
    - 学习率: 0.001
    """
    
    def __init__(self, state_dim: int = 10, action_dim: int = 4):
        """
        初始化DQN网络
        
        Args:
            state_dim: 状态维度 (双向状态特征，固定为10)
            action_dim: 动作维度 (4种组合: 00, 01, 10, 11)
        """
        super(DQNNetwork, self).__init__()
        
        layers = []
        
        # 输入层到第一个隐藏层
        layers.extend([
            nn.Linear(state_dim, 500),
            nn.ReLU()
        ])
        
        # 12个隐藏层，每层500个神经元
        for _ in range(11):  # 已经有1层，再加11层共12层
            layers.extend([
                nn.Linear(500, 500),
                nn.ReLU()
            ])
            
        # 输出层
        layers.append(nn.Linear(500, action_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class ReplayBuffer:
    """经验回放缓冲区
    
    根据论文表2-2规范:
    - 经验池大小: 3000
    """
    
    def __init__(self, capacity: int = 3000):
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
    """DRL-TSBC智能体 - 双向动态公交时刻表排班
    
    根据论文表2-2规范:
    - 学习率: 0.001
    - 折现系数gamma: 0.4
    - epsilon: 0.1 (固定值)
    - 批次大小: 64
    - 经验池大小: 3000
    - 学习频率: 5
    - 参数更新频率: 100
    """
    
    def __init__(self, 
                 state_dim: int = 10,  # 双向状态特征
                 action_dim: int = 4,   # 4种动作组合
                 learning_rate: float = 0.001,
                 gamma: float = 0.4,  # 论文规范: 0.4
                 epsilon: float = 0.1,  # 论文规范: 固定0.1
                 batch_size: int = 64,  # 论文规范: 64
                 buffer_size: int = 3000,  # 论文规范: 3000
                 learning_freq: int = 5,  # 论文规范: 每5步学习一次
                 target_update_freq: int = 100,  # 论文规范: 100
                 omega: float = 1.0/1000,  # 论文规范: 1/1000
                 zeta: float = 0.002):  # 论文规范: 0.002
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon  # 固定值，不衰减
        self.batch_size = batch_size
        self.learning_freq = learning_freq
        self.target_update_freq = target_update_freq
        
        # 神经网络
        self.q_network = DQNNetwork(state_dim, action_dim=action_dim)
        self.target_network = DQNNetwork(state_dim, action_dim=action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # 经验回放
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)
        
        # 训练计数器
        self.train_step = 0
        self.step_counter = 0  # 用于控制学习频率
        
        # 约束参数 (论文规范)
        self.t_min = 3   # 最小发车间隔(分钟) - 论文规范
        self.t_max = 15  # 最大发车间隔(分钟)
        
        # 奖励函数参数
        self.omega = omega  # 等待时间权重
        self.zeta = zeta    # 发车次数差异权重
        self.balance_threshold = 5  # 硬约束阈值
        
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
        
        根据论文方程(2.1)，状态空间包含10个维度:
        1. a1_m: 标准化小时 (th/24)
        2. a2_m: 标准化分钟 (tm/60)
        3. x1_m: 上行满载率
        4. x2_m: 上行等待时间
        5. x3_m: 上行运力利用率
        6. x4_m: 上行发车次数
        7. y1_m: 下行满载率
        8. y2_m: 下行等待时间
        9. y3_m: 下行运力利用率
        10. y4_m: 下行发车次数
        """
        features = np.array([
            # 时间特征
            env_state['time_hour'] / 24.0,           # a1_m
            env_state['time_minute'] / 60.0,         # a2_m
            
            # 上行方向特征
            env_state['up_load_factor'],             # x1_m: 上行满载率
            env_state['up_waiting_time'],            # x2_m: 上行等待时间
            env_state['up_capacity_utilization'],    # x3_m: 上行运力利用率
            env_state['up_departure_count_norm'],    # x4_m: 上行发车次数(归一化)
            
            # 下行方向特征
            env_state['down_load_factor'],           # y1_m: 下行满载率
            env_state['down_waiting_time'],          # y2_m: 下行等待时间
            env_state['down_capacity_utilization'],  # y3_m: 下行运力利用率
            env_state['down_departure_count_norm']   # y4_m: 下行发车次数(归一化)
        ])
        return features
        
    def select_action(self, state: np.ndarray, last_intervals: Dict[str, int], 
                     up_count: int = 0, down_count: int = 0) -> Tuple[int, int]:
        """
        选择动作（带双向约束的ε-贪婪策略）
        
        根据论文算法2.1，流程为：
        1. 先用DQN选择动作
        2. 然后应用发车间隔约束进行修正
        3. 应用发车次数平衡硬约束
        
        Args:
            state: 当前状态特征
            last_intervals: {'up': 上行间隔, 'down': 下行间隔}
            up_count: 当前上行发车次数
            down_count: 当前下行发车次数
            
        Returns:
            (a_up, a_down): 上行和下行的发车决策
        """
        
        # 步骤1: ε-贪婪策略选择初始动作
        if random.random() < self.epsilon:
            # 随机选择动作
            action_idx = random.randint(0, self.action_dim - 1)
        else:
            # 贪婪选择
            with torch.no_grad():
                # 确保tensor在正确的设备上
                device = next(self.q_network.parameters()).device
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = self.q_network(state_tensor).squeeze()
                action_idx = q_values.argmax().item()
        
        # 获取初始动作
        a_up, a_down = self.action_map[action_idx]
        
        # 步骤2: 应用发车间隔约束（论文算法2.1第13-20行）
        # 上行约束检查
        if last_intervals['up'] < self.t_min:
            a_up = 0  # 强制不发车
        elif last_intervals['up'] >= self.t_max:
            a_up = 1  # 强制发车
            
        # 下行约束检查  
        if last_intervals['down'] < self.t_min:
            a_down = 0  # 强制不发车
        elif last_intervals['down'] >= self.t_max:
            a_down = 1  # 强制发车
        
        # 步骤3: 应用发车次数平衡硬约束
        diff = up_count - down_count
        
        # 如果上行多发车超过阈值
        if diff > self.balance_threshold:
            if a_up == 1:  # 禁止上行发车
                a_up = 0
            # 如果下行满足发车条件，强制下行发车
            if a_down == 0 and last_intervals['down'] >= self.t_min:
                a_down = 1
        
        # 如果下行多发车超过阈值
        elif diff < -self.balance_threshold:
            if a_down == 1:  # 禁止下行发车
                a_down = 0
            # 如果上行满足发车条件，强制上行发车
            if a_up == 0 and last_intervals['up'] >= self.t_min:
                a_up = 1
                
        return (a_up, a_down)
                
    def calculate_reward(self, action: Tuple[int, int], env_state: Dict[str, Any]) -> float:
        """
        计算双向约束奖励函数（增强版：强化对等待乘客的响应）
        
        基于论文公式2.16和2.17，但增加对当前等待乘客数量的直接惩罚
        """
        a_up, a_down = action
        
        # 获取状态信息
        o_up = env_state['up_capacity_utilization']
        w_up = env_state['up_waiting_time'] 
        d_up = env_state['up_stranded_passengers']
        c_up = env_state['up_departure_count']
        
        o_down = env_state['down_capacity_utilization']
        w_down = env_state['down_waiting_time']
        d_down = env_state['down_stranded_passengers']
        c_down = env_state['down_departure_count']
        
        # 获取当前等待乘客数量（关键改进）
        waiting_up = env_state.get('up_waiting_passengers', 0)
        waiting_down = env_state.get('down_waiting_passengers', 0)
        
        # 奖励函数参数（论文规范）
        beta = 0.2    # 滞留乘客惩罚权重
        omega = self.omega  # 等待时间权重 = 1/1000
        zeta = self.zeta    # 发车次数差异权重
        
        # 新增：等待乘客惩罚权重（标准化到0-1范围）
        gamma = 0.05  # 每个等待乘客的惩罚（增加到0.05以更强烈地鼓励发车）
        
        # 计算发车次数差异
        departure_diff = c_up - c_down
        
        # 计算 o_m / c_m（平均每次发车的载客率）
        o_up_per_trip = o_up / c_up if c_up > 0 else 0
        o_down_per_trip = o_down / c_down if c_down > 0 else 0
        
        # 标准化等待乘客数量（假设最大100人）
        waiting_up_norm = min(waiting_up / 100.0, 1.0)
        waiting_down_norm = min(waiting_down / 100.0, 1.0)
        
        # 上行奖励（增强版）
        if a_up == 0:  # 不发车
            # 基础奖励 - 等待时间惩罚 - 滞留乘客惩罚 - 当前等待乘客惩罚 + 平衡奖励
            r_up = 1 - o_up_per_trip - (omega * w_up) - (beta * d_up) - (gamma * waiting_up_norm) + (zeta * departure_diff)
        else:  # 发车
            # 载客率奖励 - 滞留乘客惩罚 + 缓解等待压力奖励 - 平衡惩罚
            r_up = o_up_per_trip - (beta * d_up) + (gamma * waiting_up_norm * 0.5) - (zeta * departure_diff)
            
        # 下行奖励（增强版）
        if a_down == 0:  # 不发车
            # 基础奖励 - 等待时间惩罚 - 滞留乘客惩罚 - 当前等待乘客惩罚 - 平衡惩罚
            r_down = 1 - o_down_per_trip - (omega * w_down) - (beta * d_down) - (gamma * waiting_down_norm) - (zeta * departure_diff)
        else:  # 发车
            # 载客率奖励 - 滞留乘客惩罚 + 缓解等待压力奖励 + 平衡奖励
            r_down = o_down_per_trip - (beta * d_down) + (gamma * waiting_down_norm * 0.5) + (zeta * departure_diff)
            
        # 总奖励
        total_reward = r_up + r_down
        
        return total_reward
        
    def train(self):
        """训练智能体"""
        if len(self.replay_buffer) < self.batch_size:
            return None
            
        # 采样批次数据
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # 确保所有tensor在正确的设备上
        device = next(self.q_network.parameters()).device
        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        dones = dones.to(device)
        
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
        
        # epsilon固定为0.1，不衰减（论文规范）
        
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
    
    def balance_timetable(self, up_departures: List[int], down_departures: List[int], 
                         service_end: int) -> Tuple[List[int], List[int]]:
        """
        最终调整确保上下行发车次数相等（论文中的修正算法）
        
        Args:
            up_departures: 上行发车时间列表
            down_departures: 下行发车时间列表
            service_end: 服务结束时间
            
        Returns:
            调整后的(上行发车时间, 下行发车时间)
        """
        up_count = len(up_departures)
        down_count = len(down_departures)
        
        if up_count == down_count:
            return up_departures, down_departures
        
        # 如果上行多，删除一些上行发车
        if up_count > down_count:
            diff = up_count - down_count
            # 删除间隔最小的发车（优化服务质量）
            up_departures_sorted = sorted(up_departures)
            intervals = [up_departures_sorted[i+1] - up_departures_sorted[i] 
                        for i in range(len(up_departures_sorted)-1)]
            # 找到间隔最小的发车并删除
            to_remove = []
            for _ in range(diff):
                if len(intervals) > 0:
                    min_idx = intervals.index(min(intervals))
                    to_remove.append(up_departures_sorted[min_idx+1])
                    # 更新间隔列表
                    if min_idx < len(intervals) - 1:
                        intervals[min_idx] = up_departures_sorted[min_idx+2] - up_departures_sorted[min_idx]
                    intervals.pop(min_idx+1 if min_idx+1 < len(intervals) else min_idx)
                    up_departures_sorted.pop(min_idx+1)
            up_departures = [t for t in up_departures if t not in to_remove]
        
        # 如果下行多，删除一些下行发车
        elif down_count > up_count:
            diff = down_count - up_count
            down_departures_sorted = sorted(down_departures)
            intervals = [down_departures_sorted[i+1] - down_departures_sorted[i] 
                        for i in range(len(down_departures_sorted)-1)]
            to_remove = []
            for _ in range(diff):
                if len(intervals) > 0:
                    min_idx = intervals.index(min(intervals))
                    to_remove.append(down_departures_sorted[min_idx+1])
                    if min_idx < len(intervals) - 1:
                        intervals[min_idx] = down_departures_sorted[min_idx+2] - down_departures_sorted[min_idx]
                    intervals.pop(min_idx+1 if min_idx+1 < len(intervals) else min_idx)
                    down_departures_sorted.pop(min_idx+1)
            down_departures = [t for t in down_departures if t not in to_remove]
        
        return sorted(up_departures), sorted(down_departures)
        
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
                
                # 训练 (根据论文规范，每5步学习一次)
                self.agent.step_counter += 1
                loss = None
                if self.agent.step_counter % self.agent.learning_freq == 0:
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