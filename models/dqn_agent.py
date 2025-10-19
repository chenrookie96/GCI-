"""DQN智能体"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from typing import Tuple, List, Dict
from models.dqn_network import DQN
from models.replay_buffer import ReplayBuffer
from utils.config import DQNConfig


class DQNAgent:
    """DQN智能体"""
    
    def __init__(self, config: DQNConfig, device: str = 'cuda'):
        """
        初始化DQN智能体
        
        Args:
            config: DQN配置
            device: 设备 ('cuda' 或 'cpu')
        """
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print(f"使用设备: {self.device}")
        
        # 创建主网络和目标网络
        self.policy_net = DQN(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers
        ).to(self.device)
        
        self.target_net = DQN(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers
        ).to(self.device)
        
        # 复制主网络参数到目标网络
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # 优化器
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=config.learning_rate
        )
        
        # 超参数
        self.gamma = config.gamma
        self.epsilon = config.epsilon
        self.batch_size = config.batch_size
        self.learning_freq = config.learning_freq
        self.target_update_freq = config.target_update_freq
        
        # 经验回放池
        self.replay_buffer = ReplayBuffer(capacity=config.replay_buffer_size)
        
        # 训练统计
        self.steps = 0
        self.learn_steps = 0
        self.losses = []
        
    def select_action(self, state: np.ndarray, epsilon: float = None) -> int:
        """
        选择动作（ε-贪婪策略）
        
        Args:
            state: 当前状态
            epsilon: 探索率，如果为None则使用默认值
            
        Returns:
            动作索引 (0-3)
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        if random.random() < epsilon:
            # 随机探索
            return random.randint(0, 3)
        else:
            # 贪婪选择
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
    
    def action_index_to_tuple(self, action_idx: int) -> Tuple[int, int]:
        """
        将动作索引转换为元组
        
        Args:
            action_idx: 动作索引 (0-3)
            
        Returns:
            (a_up, a_down)
        """
        actions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        return actions[action_idx]
    
    def action_tuple_to_index(self, action: Tuple[int, int]) -> int:
        """
        将动作元组转换为索引
        
        Args:
            action: (a_up, a_down)
            
        Returns:
            动作索引 (0-3)
        """
        actions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        return actions.index(action)
    
    def store_transition(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool):
        """
        存储经验
        
        Args:
            state: 当前状态
            action: 动作索引
            reward: 奖励
            next_state: 下一状态
            done: 是否结束
        """
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def learn(self) -> float:
        """
        从经验池采样并更新网络
        
        Returns:
            损失值
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # 采样
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        
        # 转换为张量
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 计算当前Q值
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # 计算损失
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        
        self.optimizer.step()
        
        self.learn_steps += 1
        self.losses.append(loss.item())
        
        # 更新目标网络
        if self.learn_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def train_step(self, state: np.ndarray, action_tuple: Tuple[int, int],
                   reward: float, next_state: np.ndarray, done: bool) -> float:
        """
        执行一步训练
        
        Args:
            state: 当前状态
            action_tuple: 动作元组
            reward: 奖励
            next_state: 下一状态
            done: 是否结束
            
        Returns:
            损失值
        """
        # 转换动作格式
        action_idx = self.action_tuple_to_index(action_tuple)
        
        # 存储经验
        self.store_transition(state, action_idx, reward, next_state, done)
        
        self.steps += 1
        
        # 学习
        loss = 0.0
        if self.steps % self.learning_freq == 0 and len(self.replay_buffer) >= self.batch_size:
            loss = self.learn()
        
        return loss
    
    def save(self, path: str):
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps': self.steps,
            'learn_steps': self.learn_steps,
            'config': self.config
        }, path)
        print(f"模型已保存到: {path}")
    
    def load(self, path: str):
        """
        加载模型
        
        Args:
            path: 模型路径
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps = checkpoint.get('steps', 0)
        self.learn_steps = checkpoint.get('learn_steps', 0)
        print(f"模型已从 {path} 加载")
    
    def get_statistics(self) -> Dict:
        """
        获取训练统计信息
        
        Returns:
            统计信息字典
        """
        return {
            'steps': self.steps,
            'learn_steps': self.learn_steps,
            'buffer_size': len(self.replay_buffer),
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0.0
        }


if __name__ == '__main__':
    # 测试DQN智能体
    print("测试DQN智能体...")
    
    from utils.config import ConfigManager
    
    config = ConfigManager.get_default_dqn_config()
    agent = DQNAgent(config, device='cuda')
    
    print(f"\n智能体配置:")
    print(f"  状态维度: {config.state_dim}")
    print(f"  动作维度: {config.action_dim}")
    print(f"  隐藏层维度: {config.hidden_dim}")
    print(f"  隐藏层数量: {config.num_layers}")
    print(f"  学习率: {config.learning_rate}")
    print(f"  批次大小: {config.batch_size}")
    
    # 测试动作选择
    state = np.random.randn(10)
    action_idx = agent.select_action(state)
    action_tuple = agent.action_index_to_tuple(action_idx)
    print(f"\n测试动作选择:")
    print(f"  动作索引: {action_idx}")
    print(f"  动作元组: {action_tuple}")
    
    # 测试训练步骤
    next_state = np.random.randn(10)
    reward = 1.0
    done = False
    
    loss = agent.train_step(state, action_tuple, reward, next_state, done)
    print(f"\n训练步骤:")
    print(f"  损失: {loss:.4f}")
    print(f"  经验池大小: {len(agent.replay_buffer)}")
    
    print("\n测试完成！")
