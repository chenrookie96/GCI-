# experiments/tune_parameters.py
"""
参数调优脚本 - 快速测试不同的omega和zeta参数组合
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path
import json
from datetime import datetime

from src.algorithms.drl_tsbc import DRLTSBCAgent
from src.environment.bidirectional_bus_simulator import BidirectionalBusEnvironment
from src.config.bus_config import get_route_config
from src.data_processing.passenger_data_loader import PassengerDataLoader


def quick_test(omega, zeta, episodes=50, route_id='208'):
    """
    快速测试参数组合
    
    Args:
        omega: 等待时间权重
        zeta: 发车次数差异权重
        episodes: 测试轮数
        route_id: 路线编号
    
    Returns:
        dict: 测试结果统计
    """
    print(f"\n{'='*60}")
    print(f"测试参数: omega={omega:.6f}, zeta={zeta:.4f}")
    print(f"{'='*60}")
    
    # 获取配置
    config = get_route_config(route_id)
    
    # 加载数据
    data_loader = PassengerDataLoader('test_data')
    try:
        route_data = data_loader.load_route_data(route_id)
        use_real_data = True
    except:
        route_data = None
        use_real_data = False
    
    # 创建环境
    env = BidirectionalBusEnvironment(
        service_start=config.service_start,
        service_end=config.service_end,
        num_stations=config.num_stations_up,
        bus_capacity=config.capacity,
        avg_travel_time=config.avg_travel_time,
        enable_logging=False
    )
    
    if use_real_data:
        env.load_passenger_data(route_data)
    
    # 创建智能体
    agent = DRLTSBCAgent(
        state_dim=10,
        action_dim=4,
        learning_rate=0.001,
        gamma=0.4,
        epsilon=0.1,
        batch_size=64,
        buffer_size=3000,
        learning_freq=5,
        target_update_freq=100,
        omega=omega,
        zeta=zeta
    )
    
    # 移到GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        agent.q_network = agent.q_network.to(device)
        agent.target_network = agent.target_network.to(device)
    
    # 训练
    up_deps_list = []
    down_deps_list = []
    diff_list = []
    
    for episode in range(episodes):
        state = env.reset()
        last_intervals = {'up': config.tmax, 'down': config.tmax}
        episode_actions = []
        
        while not env.is_done():
            state_features = agent.get_state_features(state)
            up_count = state['up_departure_count']
            down_count = state['down_departure_count']
            
            action = agent.select_action(state_features, last_intervals, up_count, down_count)
            next_state, _, done = env.step(action)
            reward = agent.calculate_reward(action, state)
            
            next_state_features = agent.get_state_features(next_state)
            action_idx = agent.action_to_index(action)
            
            agent.replay_buffer.push(
                state_features, action_idx, reward, next_state_features, done
            )
            
            agent.step_counter += 1
            if agent.step_counter % agent.learning_freq == 0:
                agent.train()
            
            state = next_state
            episode_actions.append(action)
            
            if action[0] == 1:
                last_intervals['up'] = 0
            else:
                last_intervals['up'] += 1
                
            if action[1] == 1:
                last_intervals['down'] = 0
            else:
                last_intervals['down'] += 1
        
        up_deps = sum(1 for a in episode_actions if a[0] == 1)
        down_deps = sum(1 for a in episode_actions if a[1] == 1)
        diff = abs(up_deps - down_deps)
        
        up_deps_list.append(up_deps)
        down_deps_list.append(down_deps)
        diff_list.append(diff)
    
    # 计算统计
    last_20 = min(20, episodes)
    results = {
        'omega': omega,
        'zeta': zeta,
        'avg_up_deps': np.mean(up_deps_list[-last_20:]),
        'avg_down_deps': np.mean(down_deps_list[-last_20:]),
        'avg_total_deps': np.mean([u + d for u, d in zip(up_deps_list[-last_20:], down_deps_list[-last_20:])]),
        'avg_diff': np.mean(diff_list[-last_20:]),
        'max_diff': max(diff_list[-last_20:]),
        'min_diff': min(diff_list[-last_20:]),
        'balance_score': sum(1 for d in diff_list[-last_20:] if d <= 1) / last_20
    }
    
    print(f"\n结果 (最后{last_20}轮):")
    print(f"  平均上行发车: {results['avg_up_deps']:.1f}")
    print(f"  平均下行发车: {results['avg_down_deps']:.1f}")
    print(f"  平均总发车: {results['avg_total_deps']:.1f} (目标: ~146)")
    print(f"  平均差异: {results['avg_diff']:.2f} (目标: <1)")
    print(f"  差异范围: [{results['min_diff']}, {results['max_diff']}]")
    print(f"  平衡达标率: {results['balance_score']*100:.1f}% (差异<=1)")
    
    # 评分
    total_score = 0
    # 总发车次数评分 (目标146, 容差±10)
    total_diff = abs(results['avg_total_deps'] - 146)
    if total_diff <= 10:
        total_score += 50
    elif total_diff <= 20:
        total_score += 30
    elif total_diff <= 30:
        total_score += 10
    
    # 平衡度评分 (目标<1)
    if results['avg_diff'] < 1:
        total_score += 50
    elif results['avg_diff'] < 3:
        total_score += 30
    elif results['avg_diff'] < 5:
        total_score += 10
    
    results['score'] = total_score
    print(f"  综合评分: {total_score}/100")
    
    return results


def grid_search():
    """网格搜索最优参数"""
    print("="*80)
    print("参数网格搜索")
    print("="*80)
    
    # 参数网格
    omega_values = [1/2000, 1/3000, 1/4000, 1/5000]
    zeta_values = [0.01, 0.02, 0.03, 0.05]
    
    results = []
    
    for omega in omega_values:
        for zeta in zeta_values:
            result = quick_test(omega, zeta, episodes=50)
            results.append(result)
    
    # 排序结果
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # 保存结果
    save_dir = Path(f"results/param_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with open(save_dir / 'tuning_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 输出最佳结果
    print("\n" + "="*80)
    print("最佳参数组合 (Top 5)")
    print("="*80)
    
    for i, result in enumerate(results[:5], 1):
        print(f"\n{i}. omega={result['omega']:.6f}, zeta={result['zeta']:.4f}")
        print(f"   总发车: {result['avg_total_deps']:.1f}, 差异: {result['avg_diff']:.2f}, 评分: {result['score']}")
    
    print(f"\n结果已保存到: {save_dir / 'tuning_results.json'}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='参数调优')
    parser.add_argument('--mode', type=str, default='grid', choices=['grid', 'single'],
                       help='模式: grid=网格搜索, single=单次测试')
    parser.add_argument('--omega', type=float, default=1/3000, help='omega值 (single模式)')
    parser.add_argument('--zeta', type=float, default=0.02, help='zeta值 (single模式)')
    parser.add_argument('--episodes', type=int, default=50, help='测试轮数')
    
    args = parser.parse_args()
    
    if args.mode == 'grid':
        grid_search()
    else:
        quick_test(args.omega, args.zeta, args.episodes)
