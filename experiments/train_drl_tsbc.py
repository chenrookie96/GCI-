# experiments/train_drl_tsbc.py
"""
DRL-TSBC算法训练脚本
基于深度强化学习的双向动态公交时刻表排班算法训练
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Any
import json

from src.algorithms.drl_tsbc import DRLTSBCAgent, BidirectionalTimetableOptimizer
from src.environment.bidirectional_bus_simulator import BidirectionalBusEnvironment

def plot_training_results(history: List[Dict[str, Any]], save_path: str = None):
    """绘制训练结果"""
    episodes = [h['episode'] for h in history]
    rewards = [h['reward'] for h in history]
    epsilons = [h['epsilon'] for h in history]
    up_departures = [h['up_departures'] for h in history]
    down_departures = [h['down_departures'] for h in history]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 奖励曲线
    axes[0, 0].plot(episodes, rewards)
    axes[0, 0].set_title('Training Reward')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].grid(True)
    
    # Epsilon衰减
    axes[0, 1].plot(episodes, epsilons)
    axes[0, 1].set_title('Epsilon Decay')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Epsilon')
    axes[0, 1].grid(True)
    
    # 双向发车次数对比
    axes[1, 0].plot(episodes, up_departures, label='Up Direction', alpha=0.7)
    axes[1, 0].plot(episodes, down_departures, label='Down Direction', alpha=0.7)
    axes[1, 0].set_title('Bidirectional Departures')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Number of Departures')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 发车次数差异
    departure_diff = [abs(up - down) for up, down in zip(up_departures, down_departures)]
    axes[1, 1].plot(episodes, departure_diff)
    axes[1, 1].set_title('Departure Count Difference')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('|Up Departures - Down Departures|')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model(agent: DRLTSBCAgent, env: BidirectionalBusEnvironment, episodes: int = 10) -> Dict[str, Any]:
    """评估训练好的模型"""
    agent.epsilon = 0.0  # 关闭探索
    
    results = []
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        episode_actions = []
        last_intervals = {'up': 0, 'down': 0}
        
        while not env.is_done():
            state_features = agent.get_state_features(state)
            action = agent.select_action(state_features, last_intervals)
            
            next_state, _, done = env.step(action)
            reward = agent.calculate_reward(action, state)
            
            state = next_state
            episode_reward += reward
            episode_actions.append(action)
            
            # 更新发车间隔
            if action[0] == 1:
                last_intervals['up'] = 0
            else:
                last_intervals['up'] += 1
                
            if action[1] == 1:
                last_intervals['down'] = 0
            else:
                last_intervals['down'] += 1
        
        # 统计结果
        stats = env.get_statistics()
        results.append({
            'episode': episode,
            'reward': episode_reward,
            'up_departures': stats['up_statistics']['total_departures'],
            'down_departures': stats['down_statistics']['total_departures'],
            'departure_equal': stats['bidirectional_constraints']['departure_count_equal'],
            'up_capacity_util': stats['up_statistics']['capacity_utilization'],
            'down_capacity_util': stats['down_statistics']['capacity_utilization'],
            'up_stranded': stats['up_statistics']['stranded_passengers'],
            'down_stranded': stats['down_statistics']['stranded_passengers']
        })
    
    # 计算平均性能
    avg_results = {
        'avg_reward': np.mean([r['reward'] for r in results]),
        'avg_up_departures': np.mean([r['up_departures'] for r in results]),
        'avg_down_departures': np.mean([r['down_departures'] for r in results]),
        'departure_equality_rate': np.mean([r['departure_equal'] for r in results]),
        'avg_up_capacity_util': np.mean([r['up_capacity_util'] for r in results]),
        'avg_down_capacity_util': np.mean([r['down_capacity_util'] for r in results]),
        'avg_up_stranded': np.mean([r['up_stranded'] for r in results]),
        'avg_down_stranded': np.mean([r['down_stranded'] for r in results])
    }
    
    return avg_results, results

def main():
    """主训练函数"""
    print("开始训练DRL-TSBC算法...")
    
    # 创建环境和智能体
    env = BidirectionalBusEnvironment(
        service_start=360,   # 6:00 AM
        service_end=1320,    # 10:00 PM  
        num_stations=20,
        bus_capacity=50,
        avg_travel_time=30
    )
    
    agent = DRLTSBCAgent(
        state_dim=10,
        action_dim=4,
        learning_rate=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    optimizer = BidirectionalTimetableOptimizer(agent)
    
    # 训练参数
    training_episodes = 2000
    
    print(f"训练参数:")
    print(f"- 训练轮数: {training_episodes}")
    print(f"- 服务时间: {env.service_start//60:02d}:{env.service_start%60:02d} - {env.service_end//60:02d}:{env.service_end%60:02d}")
    print(f"- 车辆容量: {env.bus_capacity}")
    print(f"- 平均行程时间: {env.avg_travel_time}分钟")
    
    # 开始训练
    print("\n开始训练...")
    training_history = optimizer.optimize_timetable(env, episodes=training_episodes)
    
    # 保存训练历史
    os.makedirs('results', exist_ok=True)
    with open('results/drl_tsbc_training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # 绘制训练结果
    print("\n绘制训练结果...")
    plot_training_results(training_history, 'results/drl_tsbc_training_curves.png')
    
    # 保存模型
    agent.save_model('results/drl_tsbc_model.pth')
    print("模型已保存到: results/drl_tsbc_model.pth")
    
    # 评估模型
    print("\n评估训练好的模型...")
    avg_results, detailed_results = evaluate_model(agent, env, episodes=20)
    
    print("\n=== 评估结果 ===")
    print(f"平均奖励: {avg_results['avg_reward']:.2f}")
    print(f"平均上行发车次数: {avg_results['avg_up_departures']:.1f}")
    print(f"平均下行发车次数: {avg_results['avg_down_departures']:.1f}")
    print(f"发车次数相等率: {avg_results['departure_equality_rate']:.1%}")
    print(f"上行运力利用率: {avg_results['avg_up_capacity_util']:.1%}")
    print(f"下行运力利用率: {avg_results['avg_down_capacity_util']:.1%}")
    print(f"上行滞留乘客: {avg_results['avg_up_stranded']:.1f}")
    print(f"下行滞留乘客: {avg_results['avg_down_stranded']:.1f}")
    
    # 保存评估结果
    with open('results/drl_tsbc_evaluation.json', 'w') as f:
        json.dump({
            'average_results': avg_results,
            'detailed_results': detailed_results
        }, f, indent=2)
    
    print("\n训练完成！结果已保存到 results/ 目录")

if __name__ == "__main__":
    main()
