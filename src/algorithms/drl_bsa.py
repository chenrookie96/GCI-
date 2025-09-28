"""
DRL-BSA: Deep Reinforcement Learning-based Bus Scheduling Algorithm

基于深度强化学习的公交车辆调度算法
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Tuple, List, Dict, Any, Optional

class DuelingDQN(nn.Module):
    """竞争深度双Q网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(DuelingDQN, self).__init__()
        
        # 共享特征提取层
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 价值流 (Value Stream)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 优势流 (Advantage Stream)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
    
    def forward(self, x):
        features = self.feature_layer(x)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values

class DRLBSAAgent:
    """DRL-BSA算法智能体"""
    
    def __init__(self, state_dim: int, action_dim: int, 
                 learning_rate: float = 0.001, gamma: float = 0.95,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01, memory_size: int = 10000,
                 batch_size: int = 32, target_update: int = 100):
        """
        初始化DRL-BSA智能体
        
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            learning_rate: 学习率
            gamma: 折扣因子
            epsilon: 探索率
            epsilon_decay: 探索率衰减
            epsilon_min: 最小探索率
            memory_size: 经验回放缓冲区大小
            batch_size: 批次大小
            target_update: 目标网络更新频率
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update = target_update
        
        # 竞争深度双Q网络
        self.q_network = DuelingDQN(state_dim, action_dim)
        self.target_network = DuelingDQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # 经验回放
        self.memory = deque(maxlen=memory_size)
        
        # 训练步数
        self.training_steps = 0
        
        # 更新目标网络
        self.update_target_network()
    
    def update_target_network(self):
        """更新目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def get_state(self, env_state: Dict[str, Any]) -> np.ndarray:
        """
        构建状态向量
        
        Args:
            env_state: 环境状态信息
            
        Returns:
            状态向量
        """
        # 车辆状态特征
        vehicle_states = []
        
        for vehicle in env_state['vehicles']:
            # 剩余工作时间 (归一化)
            remaining_work_time = vehicle['remaining_work_time'] / env_state['max_work_time']
            
            # 剩余驾驶时间 (归一化)
            remaining_drive_time = vehicle['remaining_drive_time'] / env_state['max_drive_time']
            
            # 休息时间 (归一化)
            rest_time = vehicle['rest_time'] / env_state['max_rest_time']
            
            # 已执行行程数 (归一化)
            executed_trips = vehicle['executed_trips'] / env_state['max_trips']
            
            # 车辆类型 (one-hot编码)
            vehicle_type = [0.0] * 3  # 假设有3种车辆类型
            vehicle_type[vehicle['type']] = 1.0
            
            vehicle_state = [remaining_work_time, remaining_drive_time, 
                           rest_time, executed_trips] + vehicle_type
            vehicle_states.extend(vehicle_state)
        
        # 当前任务信息
        current_task = env_state['current_task']
        task_features = [
            current_task['departure_time'] / 1440.0,  # 发车时间 (归一化)
            current_task['duration'] / 120.0,  # 任务时长 (归一化)
            current_task['priority']  # 任务优先级
        ]
        
        # 环境状态
        env_features = [
            env_state['current_time'] / 1440.0,  # 当前时间
            env_state['traffic_congestion_level'],  # 交通拥堵程度
            env_state['weather_condition'] / 5.0  # 天气条件 (归一化)
        ]
        
        # 组合所有特征
        state_vector = vehicle_states + task_features + env_features
        
        # 确保状态向量长度一致
        if len(state_vector) < self.state_dim:
            state_vector.extend([0.0] * (self.state_dim - len(state_vector)))
        elif len(state_vector) > self.state_dim:
            state_vector = state_vector[:self.state_dim]
        
        return np.array(state_vector, dtype=np.float32)
    
    def get_valid_actions(self, env_state: Dict[str, Any]) -> List[int]:
        """
        获取有效动作列表
        
        Args:
            env_state: 环境状态
            
        Returns:
            有效动作索引列表
        """
        valid_actions = []
        
        for i, vehicle in enumerate(env_state['vehicles']):
            # 检查车辆是否满足约束条件
            if self._is_vehicle_available(vehicle, env_state):
                valid_actions.append(i)
        
        return valid_actions
    
    def _is_vehicle_available(self, vehicle: Dict[str, Any], env_state: Dict[str, Any]) -> bool:
        """
        检查车辆是否可用
        
        Args:
            vehicle: 车辆信息
            env_state: 环境状态
            
        Returns:
            车辆是否可用
        """
        # 检查工作时间约束
        if vehicle['remaining_work_time'] < env_state['current_task']['duration']:
            return False
        
        # 检查休息时间约束
        if vehicle['rest_time'] < env_state['min_rest_time']:
            return False
        
        # 检查位置约束
        if not self._check_location_constraint(vehicle, env_state):
            return False
        
        return True
    
    def _check_location_constraint(self, vehicle: Dict[str, Any], env_state: Dict[str, Any]) -> bool:
        """检查位置约束"""
        # 简化版本：假设所有车辆都可以到达任何位置
        return True
    
    def choose_action(self, state: np.ndarray, valid_actions: List[int]) -> int:
        """
        选择动作
        
        Args:
            state: 当前状态
            valid_actions: 有效动作列表
            
        Returns:
            选择的动作索引
        """
        if not valid_actions:
            return 0  # 默认动作
        
        if random.random() <= self.epsilon:
            return random.choice(valid_actions)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        
        # 只考虑有效动作
        valid_q_values = q_values[0][valid_actions]
        best_action_idx = valid_q_values.argmax().item()
        
        return valid_actions[best_action_idx]
    
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
        
        # 当前Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # 目标Q值
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # 计算损失
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # 更新目标网络
        self.training_steps += 1
        if self.training_steps % self.target_update == 0:
            self.update_target_network()

class DRLBSAEnvironment:
    """DRL-BSA仿真环境"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化环境
        
        Args:
            config: 环境配置参数
        """
        self.config = config
        self.current_time = 0
        self.vehicles = self._initialize_vehicles()
        self.tasks = self._load_tasks()
        self.current_task_idx = 0
        
    def _initialize_vehicles(self) -> List[Dict[str, Any]]:
        """初始化车辆"""
        vehicles = []
        for i in range(self.config['num_vehicles']):
            vehicle = {
                'id': i,
                'type': random.randint(0, 2),  # 车辆类型
                'remaining_work_time': self.config['max_work_time'],
                'remaining_drive_time': self.config['max_drive_time'],
                'rest_time': 0,
                'executed_trips': 0,
                'location': 0,  # 当前位置
                'status': 'available'  # 车辆状态
            }
            vehicles.append(vehicle)
        return vehicles
    
    def _load_tasks(self) -> List[Dict[str, Any]]:
        """加载任务列表"""
        tasks = []
        for i in range(self.config['num_tasks']):
            task = {
                'id': i,
                'departure_time': random.randint(0, 1440),  # 发车时间
                'duration': random.randint(30, 120),  # 任务时长
                'priority': random.uniform(0.5, 1.0),  # 优先级
                'route_id': random.randint(0, 10)  # 线路ID
            }
            tasks.append(task)
        
        # 按发车时间排序
        tasks.sort(key=lambda x: x['departure_time'])
        return tasks
    
    def reset(self) -> Dict[str, Any]:
        """重置环境"""
        self.current_time = 0
        self.current_task_idx = 0
        self.vehicles = self._initialize_vehicles()
        return self._get_state()
    
    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        执行一步动作
        
        Args:
            action: 选择的车辆索引
            
        Returns:
            (next_state, reward, done, info)
        """
        if self.current_task_idx >= len(self.tasks):
            return self._get_state(), 0.0, True, {}
        
        current_task = self.tasks[self.current_task_idx]
        
        # 分配车辆执行任务
        if action < len(self.vehicles):
            vehicle = self.vehicles[action]
            vehicle['executed_trips'] += 1
            vehicle['remaining_work_time'] -= current_task['duration']
            vehicle['remaining_drive_time'] -= current_task['duration']
            vehicle['rest_time'] = 0  # 重置休息时间
            vehicle['status'] = 'working'
        
        # 计算奖励
        reward = self._calculate_reward(action, current_task)
        
        # 更新环境状态
        self._update_environment()
        
        # 移动到下一个任务
        self.current_task_idx += 1
        
        # 获取新状态
        next_state = self._get_state()
        
        # 判断是否结束
        done = self.current_task_idx >= len(self.tasks)
        
        info = {
            'task_completed': self.current_task_idx,
            'total_tasks': len(self.tasks),
            'vehicles_used': len([v for v in self.vehicles if v['executed_trips'] > 0])
        }
        
        return next_state, reward, done, info
    
    def _update_environment(self):
        """更新环境状态"""
        # 更新车辆状态
        for vehicle in self.vehicles:
            if vehicle['status'] == 'working':
                # 检查是否完成当前任务
                if vehicle['remaining_drive_time'] <= 0:
                    vehicle['status'] = 'resting'
                    vehicle['rest_time'] = self.config['min_rest_time']
            elif vehicle['status'] == 'resting':
                vehicle['rest_time'] -= 1
                if vehicle['rest_time'] <= 0:
                    vehicle['status'] = 'available'
        
        # 更新时间
        self.current_time += 1
    
    def _get_state(self) -> Dict[str, Any]:
        """获取当前状态"""
        if self.current_task_idx >= len(self.tasks):
            current_task = {'departure_time': 0, 'duration': 0, 'priority': 0}
        else:
            current_task = self.tasks[self.current_task_idx]
        
        return {
            'vehicles': self.vehicles,
            'current_task': current_task,
            'current_time': self.current_time,
            'traffic_congestion_level': random.uniform(0.0, 1.0),
            'weather_condition': random.randint(1, 5),
            'max_work_time': self.config['max_work_time'],
            'max_drive_time': self.config['max_drive_time'],
            'max_rest_time': self.config['max_rest_time'],
            'max_trips': self.config['max_trips'],
            'min_rest_time': self.config['min_rest_time']
        }
    
    def _calculate_reward(self, action: int, task: Dict[str, Any]) -> float:
        """计算奖励"""
        reward = 0.0
        
        if action < len(self.vehicles):
            vehicle = self.vehicles[action]
            
            # 基础奖励：成功分配车辆
            reward += 1.0
            
            # 效率奖励：优先使用执行次数少的车辆
            if vehicle['executed_trips'] == 0:
                reward += 0.5  # 使用新车辆奖励
            
            # 平衡奖励：鼓励所有车辆都有偶数次发车
            if vehicle['executed_trips'] % 2 == 0:
                reward += 0.2
            
            # 任务优先级奖励
            reward += task['priority'] * 0.3
            
        else:
            # 无效动作惩罚
            reward -= 1.0
        
        return reward

def train_drl_bsa(episodes: int = 1000, save_path: str = "models/drl_bsa.pth"):
    """训练DRL-BSA模型"""
    
    # 环境配置
    config = {
        'num_vehicles': 10,
        'num_tasks': 100,
        'max_work_time': 480,  # 8小时
        'max_drive_time': 360,  # 6小时
        'max_rest_time': 60,   # 1小时
        'max_trips': 20,
        'min_rest_time': 15    # 15分钟
    }
    
    # 创建环境和智能体
    env = DRLBSAEnvironment(config)
    agent = DRLBSAAgent(state_dim=50, action_dim=config['num_vehicles'])
    
    # 训练循环
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        while True:
            # 获取状态向量
            state_vector = agent.get_state(state)
            valid_actions = agent.get_valid_actions(state)
            
            if not valid_actions:
                break
            
            # 选择动作
            action = agent.choose_action(state_vector, valid_actions)
            
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
        
        # 打印训练进度
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}, Vehicles Used: {info.get('vehicles_used', 0)}")
    
    # 保存模型
    torch.save(agent.q_network.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_drl_bsa()
