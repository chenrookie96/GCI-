"""辅助工具函数"""
import torch
import numpy as np
import random
import os
import json
from typing import Dict, Any


def set_seed(seed: int):
    """
    设置所有随机种子以确保可复现性
    
    Args:
        seed: 随机种子
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # 设置PyTorch的确定性行为
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"随机种子已设置为: {seed}")


def check_cuda():
    """检查CUDA是否可用"""
    if torch.cuda.is_available():
        print(f"CUDA可用")
        print(f"  设备数量: {torch.cuda.device_count()}")
        print(f"  当前设备: {torch.cuda.current_device()}")
        print(f"  设备名称: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA版本: {torch.version.cuda}")
        return True
    else:
        print("CUDA不可用，将使用CPU")
        return False


def save_json(data: Dict[str, Any], filepath: str):
    """
    保存数据到JSON文件
    
    Args:
        data: 要保存的数据
        filepath: 文件路径
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"数据已保存到: {filepath}")


def load_json(filepath: str) -> Dict[str, Any]:
    """
    从JSON文件加载数据
    
    Args:
        filepath: 文件路径
        
    Returns:
        加载的数据
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def format_time(minutes: int) -> str:
    """
    将分钟数格式化为HH:MM格式
    
    Args:
        minutes: 分钟数
        
    Returns:
        格式化的时间字符串
    """
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours:02d}:{mins:02d}"


def parse_time(time_str: str) -> int:
    """
    将HH:MM格式的时间解析为分钟数
    
    Args:
        time_str: 时间字符串 (HH:MM)
        
    Returns:
        分钟数
    """
    parts = time_str.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    return hours * 60 + minutes


def create_directories():
    """创建所有必需的目录"""
    directories = [
        'results/models',
        'results/logs',
        'results/figures',
        'results/tables',
        'configs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("目录结构已创建")


def print_model_summary(model):
    """
    打印模型摘要
    
    Args:
        model: PyTorch模型
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("模型摘要:")
    print(f"  总参数数: {total_params:,}")
    print(f"  可训练参数数: {trainable_params:,}")
    print(f"  模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    获取计算设备
    
    Args:
        prefer_cuda: 是否优先使用CUDA
        
    Returns:
        torch.device对象
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"使用设备: {device} ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device('cpu')
        print(f"使用设备: {device}")
    
    return device


def count_files_in_directory(directory: str, extension: str = None) -> int:
    """
    统计目录中的文件数量
    
    Args:
        directory: 目录路径
        extension: 文件扩展名（可选）
        
    Returns:
        文件数量
    """
    if not os.path.exists(directory):
        return 0
    
    files = os.listdir(directory)
    
    if extension:
        files = [f for f in files if f.endswith(extension)]
    
    return len(files)


class ProgressTracker:
    """训练进度跟踪器"""
    
    def __init__(self, total_episodes: int):
        self.total_episodes = total_episodes
        self.current_episode = 0
        self.best_reward = float('-inf')
        self.episode_rewards = []
    
    def update(self, episode: int, reward: float):
        """更新进度"""
        self.current_episode = episode
        self.episode_rewards.append(reward)
        
        if reward > self.best_reward:
            self.best_reward = reward
    
    def get_progress(self) -> float:
        """获取进度百分比"""
        return (self.current_episode / self.total_episodes) * 100
    
    def get_average_reward(self, last_n: int = 10) -> float:
        """获取最近N个episode的平均奖励"""
        if len(self.episode_rewards) == 0:
            return 0.0
        
        recent_rewards = self.episode_rewards[-last_n:]
        return np.mean(recent_rewards)
    
    def print_summary(self):
        """打印摘要"""
        print(f"\n进度: {self.get_progress():.1f}%")
        print(f"当前episode: {self.current_episode}/{self.total_episodes}")
        print(f"最佳奖励: {self.best_reward:.2f}")
        print(f"最近10个episode平均奖励: {self.get_average_reward():.2f}")


if __name__ == '__main__':
    print("辅助工具模块测试")
    print("-" * 50)
    
    # 测试CUDA
    check_cuda()
    
    # 测试时间格式化
    print(f"\n时间格式化测试:")
    print(f"  360分钟 = {format_time(360)}")
    print(f"  1260分钟 = {format_time(1260)}")
    print(f"  '06:00' = {parse_time('06:00')}分钟")
    
    # 创建目录
    print(f"\n创建目录:")
    create_directories()
    
    print("\n测试完成！")
