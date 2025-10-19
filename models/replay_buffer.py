"""经验回放池"""
import random
import numpy as np
from collections import deque
from typing import Tuple, List


class ReplayBuffer:
    """经验回放池"""
    
    def __init__(self, capacity: int = 3000):
        """
        初始化经验回放池
        
        Args:
            capacity: 经验池容量
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        """
        添加经验
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一状态
            done: 是否结束
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                np.ndarray, np.ndarray]:
        """
        随机采样一批经验
        
        Args:
            batch_size: 批次大小
            
        Returns:
            (states, actions, rewards, next_states, dones)
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self) -> int:
        """返回当前经验池大小"""
        return len(self.buffer)
    
    def is_full(self) -> bool:
        """判断经验池是否已满"""
        return len(self.buffer) >= self.capacity
    
    def clear(self):
        """清空经验池"""
        self.buffer.clear()


if __name__ == '__main__':
    # 测试经验回放池
    print("测试经验回放池...")
    
    buffer = ReplayBuffer(capacity=100)
    
    # 添加一些经验
    for i in range(150):
        state = np.random.randn(10)
        action = random.randint(0, 3)
        reward = random.random()
        next_state = np.random.randn(10)
        done = random.random() > 0.9
        
        buffer.push(state, action, reward, next_state, done)
    
    print(f"经验池大小: {len(buffer)} (容量: {buffer.capacity})")
    print(f"是否已满: {buffer.is_full()}")
    
    # 采样
    batch_size = 32
    states, actions, rewards, next_states, dones = buffer.sample(batch_size)
    
    print(f"\n采样批次大小: {batch_size}")
    print(f"states形状: {states.shape}")
    print(f"actions形状: {actions.shape}")
    print(f"rewards形状: {rewards.shape}")
    print(f"next_states形状: {next_states.shape}")
    print(f"dones形状: {dones.shape}")
