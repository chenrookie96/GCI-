# experiments/evaluate_drl_tsbc.py
"""
DRL-TSBC评估脚本
实现推理、评估和性能对比
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

from src.algorithms.drl_tsbc import DRLTSBCAgent
from src.environment.bidirectional_bus_simulator import BidirectionalBusEnvironment
from src.config.bus_config import get_route_config


class TimetableBalancer:
    """
    时刻表平衡调整器
    
    根据论文算法2.2实现推理阶段的发车次数平衡
    """
    
    def __init__(self, tmax: int = 15):
        """
        初始化平衡调整器
        
        Args:
            tmax: 最大发车间隔（分钟）
        """
        self.tmax = tmax
    
    def balance_timetable(self, timetable: Dict[str, List[int]]) -> Dict[str, List[int]]:
        """
        平衡双向时刻表
        
        根据论文算法2.2:
        1. 如果上下行发车次数相等，直接返回
        2. 删除发车次数多的方向的倒数第二次发车
        3. 以Tmax为间隔向前调整后续发车时刻
        
        Args:
            timetable: {'up': [发车时刻列表], 'down': [发车时刻列表]}
            
        Returns:
            平衡后的时刻表
        """
        up_times = timetable['up'].copy()
        down_times = timetable['down'].copy()
        
        # 检查发车次数
        up_count = len(up_times)
        down_count = len(down_times)
        
        if up_count == down_count:
            return {'up': up_times, 'down': down_times}
        
        # 确定需要调整的方向
        if up_count > down_count:
            direction = 'up'
            times = up_times
        else:
            direction = 'down'
            times = down_times
        
        # 删除倒数第二次发车
        if len(times) >= 2:
            removed_time = times[-2]
            times.pop(-2)
            
            # 以Tmax为间隔向前调整后续发车
            last_time = times[-1]
            adjusted_time = removed_time + self.tmax
            
            # 如果调整后的时间不超过最后一次发车，则更新
            if adjusted_time <= last_time:
                times[-1] = adjusted_time
        
        # 返回平衡后的时刻表
        if direction == 'up':
            return {'up': times, 'down': down_times}
        else:
            return {'up': up_times, 'down': times}


class PerformanceEvaluator:
    """性能评估器"""
    
    def __init__(self, env: BidirectionalBusEnvironment):
        self.env = env
    
    def evaluate_timetable(self, timetable: Dict[str, List[int]]) -> Dict[str, float]:
        """
        评估时刻表性能
        
        Args:
            timetable: 时刻表
            
        Returns:
            性能指标字典
        """
        # 重置环境
        state = self.env.reset()
        
        # 将时刻表转换为动作序列
        up_schedule = set(timetable['up'])
        down_schedule = set(timetable['down'])
        
        total_waiting_time = 0
        total_passengers = 0
        stranded_passengers = 0
        
        while not self.env.is_done():
            current_time = self.env.current_time
            
            # 根据时刻表决定动作
            a_up = 1 if current_time in up_schedule else 0
            a_down = 1 if current_time in down_schedule else 0
            action = (a_up, a_down)
            
            # 执行动作
            next_state, _, done = self.env.step(action)
            state = next_state
        
        # 获取统计信息
        stats = self.env.get_statistics()
        
        # 计算性能指标
        up_stats = stats['up_statistics']
        down_stats = stats['down_statistics']
        
        metrics = {
            'avg_waiting_time': (up_stats['total_waiting_time'] + down_stats['total_waiting_time']) / 
                               (up_stats['total_passengers_served'] + down_stats['total_passengers_served'] + 1e-6),
            'total_departures': up_stats['total_departures'] + down_stats['total_departures'],
            'up_departures': up_stats['total_departures'],
            'down_departures': down_stats['total_departures'],
            'departure_balance': abs(up_stats['total_departures'] - down_stats['total_departures']),
            'total_passengers_served': up_stats['total_passengers_served'] + down_stats['total_passengers_served'],
            'stranded_passengers': up_stats['stranded_passengers'] + down_stats['stranded_passengers'],
            'capacity_utilization': (up_stats['capacity_utilization'] + down_stats['capacity_utilization']) / 2
        }
        
        return metrics


def generate_timetable_from_agent(agent: DRLTSBCAgent, 
                                  env: BidirectionalBusEnvironment,
                                  balance: bool = True) -> Dict[str, List[int]]:
    """
    使用训练好的智能体生成时刻表
    
    Args:
        agent: 训练好的DRL-TSBC智能体
        env: 仿真环境
        balance: 是否进行平衡调整
        
    Returns:
        时刻表字典
    """
    state = env.reset()
    up_times = []
    down_times = []
    last_intervals = {'up': 0, 'down': 0}
    
    while not env.is_done():
        # 获取状态特征
        state_features = agent.get_state_features(state)
        
        # 选择动作（使用贪婪策略）
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_features).unsqueeze(0)
            q_values = agent.q_network(state_tensor).squeeze()
            
            # 应用约束
            masked_q_values = q_values.clone()
            
            # 上行约束
            if last_intervals['up'] < agent.t_min:
                for action_idx, (a_up, a_down) in agent.action_map.items():
                    if a_up == 1:
                        masked_q_values[action_idx] = float('-inf')
            elif last_intervals['up'] >= agent.t_max:
                for action_idx, (a_up, a_down) in agent.action_map.items():
                    if a_up == 0:
                        masked_q_values[action_idx] = float('-inf')
            
            # 下行约束
            if last_intervals['down'] < agent.t_min:
                for action_idx, (a_up, a_down) in agent.action_map.items():
                    if a_down == 1:
                        masked_q_values[action_idx] = float('-inf')
            elif last_intervals['down'] >= agent.t_max:
                for action_idx, (a_up, a_down) in agent.action_map.items():
                    if a_down == 0:
                        masked_q_values[action_idx] = float('-inf')
            
            action_idx = masked_q_values.argmax().item()
            action = agent.action_map[action_idx]
        
        # 记录发车时刻
        if action[0] == 1:
            up_times.append(env.current_time)
            last_intervals['up'] = 0
        else:
            last_intervals['up'] += 1
        
        if action[1] == 1:
            down_times.append(env.current_time)
            last_intervals['down'] = 0
        else:
            last_intervals['down'] += 1
        
        # 执行动作
        next_state, _, done = env.step(action)
        state = next_state
    
    timetable = {'up': up_times, 'down': down_times}
    
    # 平衡调整
    if balance:
        balancer = TimetableBalancer()
        timetable = balancer.balance_timetable(timetable)
    
    return timetable


def evaluate_model(model_path: str, route_id: str = '208'):
    """
    评估训练好的模型
    
    Args:
        model_path: 模型文件路径
        route_id: 路线编号
    """
    print("=" * 80)
    print("DRL-TSBC 模型评估")
    print("=" * 80)
    
    # 获取路线配置
    config = get_route_config(route_id)
    print(f"\n路线: {config.route_id}")
    
    # 创建环境
    env = BidirectionalBusEnvironment(
        service_start=config.service_start,
        service_end=config.service_end,
        num_stations=config.num_stations_up,
        bus_capacity=config.capacity,
        avg_travel_time=config.avg_travel_time
    )
    
    # 加载模型
    agent = DRLTSBCAgent()
    agent.load_model(model_path)
    agent.epsilon = 0.0  # 评估时使用贪婪策略
    print(f"模型已加载: {model_path}")
    
    # 生成时刻表（不平衡）
    print("\n生成时刻表（推理阶段，不平衡）...")
    timetable_unbalanced = generate_timetable_from_agent(agent, env, balance=False)
    print(f"  上行发车次数: {len(timetable_unbalanced['up'])}")
    print(f"  下行发车次数: {len(timetable_unbalanced['down'])}")
    print(f"  发车次数差: {abs(len(timetable_unbalanced['up']) - len(timetable_unbalanced['down']))}")
    
    # 生成时刻表（平衡）
    print("\n生成时刻表（推理阶段，平衡调整）...")
    timetable_balanced = generate_timetable_from_agent(agent, env, balance=True)
    print(f"  上行发车次数: {len(timetable_balanced['up'])}")
    print(f"  下行发车次数: {len(timetable_balanced['down'])}")
    print(f"  发车次数差: {abs(len(timetable_balanced['up']) - len(timetable_balanced['down']))}")
    
    # 评估性能
    evaluator = PerformanceEvaluator(env)
    
    print("\n评估性能（不平衡）...")
    metrics_unbalanced = evaluator.evaluate_timetable(timetable_unbalanced)
    
    print("\n评估性能（平衡）...")
    metrics_balanced = evaluator.evaluate_timetable(timetable_balanced)
    
    # 输出对比结果
    print("\n" + "=" * 80)
    print("性能对比")
    print("=" * 80)
    
    comparison = pd.DataFrame({
        '不平衡': metrics_unbalanced,
        '平衡调整': metrics_balanced
    }).T
    
    print(comparison.to_string())
    
    return timetable_balanced, metrics_balanced


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='评估DRL-TSBC模型')
    parser.add_argument('--model', type=str, required=True, help='模型文件路径')
    parser.add_argument('--route', type=str, default='208', help='路线编号')
    
    args = parser.parse_args()
    
    timetable, metrics = evaluate_model(args.model, args.route)
    
    print("\n评估完成！")
