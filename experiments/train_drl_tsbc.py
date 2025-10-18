# experiments/train_drl_tsbc.py
"""
DRL-TSBC训练脚本
完整的端到端训练流程
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm

from src.algorithms.drl_tsbc import DRLTSBCAgent, BidirectionalTimetableOptimizer
from src.environment.station_level_simulator import StationLevelBusEnvironment
from src.config.bus_config import get_route_config
from src.data_processing.passenger_data_loader import PassengerDataLoader


class TrainingMonitor:
    """训练监控器（增强版）"""
    
    def __init__(self, save_dir: str = "results"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.episode_rewards = []
        self.episode_losses = []
        self.up_departures = []
        self.down_departures = []
        self.epsilon_history = []
        self.departure_differences = []  # 发车次数差异
        self.balance_violations = []     # 平衡约束违反的episode
        
    def log_episode(self, episode: int, reward: float, loss: float, 
                   up_deps: int, down_deps: int, epsilon: float):
        """记录一个episode的数据"""
        self.episode_rewards.append(reward)
        self.episode_losses.append(loss)
        self.up_departures.append(up_deps)
        self.down_departures.append(down_deps)
        self.epsilon_history.append(epsilon)
        
        # 记录发车次数差异
        diff = abs(up_deps - down_deps)
        self.departure_differences.append(diff)
        
        # 记录平衡约束违反
        if diff > 1:
            self.balance_violations.append(episode)
        
    def plot_training_curves(self, save_path: str = None):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 奖励曲线
        axes[0, 0].plot(self.episode_rewards)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].grid(True)
        
        # 损失曲线
        axes[0, 1].plot(self.episode_losses)
        axes[0, 1].set_title('Training Loss')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True)
        
        # 发车次数差异曲线
        axes[0, 2].plot(self.departure_differences, color='red')
        axes[0, 2].axhline(y=1, color='green', linestyle='--', label='Target (<=1)')
        axes[0, 2].set_title('Departure Balance (|Up - Down|)')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Difference')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # 发车次数对比
        axes[1, 0].plot(self.up_departures, label='Up', alpha=0.7)
        axes[1, 0].plot(self.down_departures, label='Down', alpha=0.7)
        axes[1, 0].axhline(y=73, color='red', linestyle='--', label='Target (73)')
        axes[1, 0].set_title('Bidirectional Departures')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Departure Count')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 总发车次数
        total_deps = [u + d for u, d in zip(self.up_departures, self.down_departures)]
        axes[1, 1].plot(total_deps, color='purple')
        axes[1, 1].axhline(y=146, color='red', linestyle='--', label='Target (~146)')
        axes[1, 1].set_title('Total Departures')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Total Count')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # 平衡度散点图
        axes[1, 2].scatter(self.up_departures, self.down_departures, alpha=0.3, s=10)
        max_val = max(max(self.up_departures), max(self.down_departures))
        axes[1, 2].plot([0, max_val], [0, max_val], 'r--', label='Perfect Balance')
        axes[1, 2].set_title('Balance Scatter Plot')
        axes[1, 2].set_xlabel('Up Departures')
        axes[1, 2].set_ylabel('Down Departures')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"训练曲线已保存到: {save_path}")
        else:
            plt.savefig(self.save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        
        plt.close()
        
    def save_metrics(self, filepath: str = None):
        """保存训练指标"""
        if filepath is None:
            filepath = self.save_dir / 'training_metrics.json'
        
        metrics = {
            'episode_rewards': self.episode_rewards,
            'episode_losses': self.episode_losses,
            'up_departures': self.up_departures,
            'down_departures': self.down_departures,
            'epsilon_history': self.epsilon_history,
            'final_avg_reward': np.mean(self.episode_rewards[-50:]) if len(self.episode_rewards) >= 50 else np.mean(self.episode_rewards),
            'final_departure_balance': abs(np.mean(self.up_departures[-50:]) - np.mean(self.down_departures[-50:])) if len(self.up_departures) >= 50 else 0
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        print(f"训练指标已保存到: {filepath}")


def train_drl_tsbc(route_id: str = '208',
                   episodes: int = 500,
                   save_interval: int = 50,
                   device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """
    训练DRL-TSBC模型
    
    Args:
        route_id: 路线编号
        episodes: 训练轮数
        save_interval: 模型保存间隔
        device: 训练设备
    """
    print("=" * 80)
    print("DRL-TSBC 训练开始")
    print("=" * 80)
    
    # 获取路线配置
    config = get_route_config(route_id)
    print(f"\n路线配置:")
    print(f"  路线: {config.route_id}")
    print(f"  服务时间: {config.service_start//60:02d}:{config.service_start%60:02d} - "
          f"{config.service_end//60:02d}:{config.service_end%60:02d}")
    print(f"  车辆容量: {config.capacity}人")
    print(f"  发车间隔约束: Tmin={config.tmin}分钟, Tmax={config.tmax}分钟")
    
    # 加载真实乘客数据
    print(f"\n加载真实乘客数据...")
    data_loader = PassengerDataLoader('test_data')
    try:
        route_data = data_loader.load_route_data(route_id)
        passenger_data = route_data['passenger']
        print(f"  成功加载{route_id}路线数据")
        print(f"  上行乘客数: {len(passenger_data['direction_0'])}")
        print(f"  下行乘客数: {len(passenger_data['direction_1'])}")
        use_real_data = True
    except Exception as e:
        print(f"  警告: 无法加载真实数据 ({e})，将使用模拟数据")
        route_data = None
        use_real_data = False
    
    # 创建环境（使用站点级别模拟器，包含固定首末班车功能）
    env = StationLevelBusEnvironment(
        service_start=config.service_start,
        service_end=config.service_end,
        num_stations=config.num_stations_up,
        bus_capacity=config.capacity,
        avg_travel_time=config.avg_travel_time,
        enable_logging=False
    )
    
    # 加载真实数据到环境
    if use_real_data:
        env.load_passenger_data(route_data)
        # 统计真实数据
        total_up = sum(len(arrivals.get('up', [])) for arrivals in env.passenger_arrivals.values())
        total_down = sum(len(arrivals.get('down', [])) for arrivals in env.passenger_arrivals.values())
        print(f"  真实数据已加载: 上行{total_up}人, 下行{total_down}人, 总计{total_up+total_down}人")
    
    # 创建智能体
    # 保持omega=1/1000符合论文规范，通过增强zeta和修正奖励函数实现平衡
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
        omega=1.0/1000,  # 论文规范: 1/1000
        zeta=0.002       # 论文规范: 0.002
    )
    
    # 将网络移到GPU
    if device == 'cuda':
        agent.q_network = agent.q_network.to(device)
        agent.target_network = agent.target_network.to(device)
        print(f"\n使用设备: {device}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"\n使用设备: CPU")
    
    # 创建优化器
    optimizer = BidirectionalTimetableOptimizer(agent)
    
    # 创建监控器
    monitor = TrainingMonitor(save_dir=f"results/{route_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # 训练循环
    print(f"\n开始训练 {episodes} 个episodes...")
    print("=" * 80)
    
    for episode in tqdm(range(episodes), desc="训练进度"):
        state = env.reset()
        episode_reward = 0
        episode_loss = 0
        loss_count = 0
        episode_actions = []
        # 初始化间隔为Tmax，确保首班车会发车
        last_intervals = {'up': config.tmax, 'down': config.tmax}
        
        while not env.is_done():
            # 获取状态特征
            state_features = agent.get_state_features(state)
            
            # 获取当前发车次数
            up_count = state['up_departure_count']
            down_count = state['down_departure_count']
            
            # 选择动作（传入发车次数用于硬约束）
            action = agent.select_action(state_features, last_intervals, up_count, down_count)
            
            # 执行动作
            next_state, _, done = env.step(action)
            
            # 计算奖励
            reward = agent.calculate_reward(action, state)
            
            # 存储经验
            next_state_features = agent.get_state_features(next_state)
            action_idx = agent.action_to_index(action)
            
            # 转换为tensor并移到设备
            if device == 'cuda':
                state_tensor = torch.FloatTensor(state_features).to(device)
                next_state_tensor = torch.FloatTensor(next_state_features).to(device)
            
            agent.replay_buffer.push(
                state_features, action_idx, reward, next_state_features, done
            )
            
            # 训练 (每5步学习一次)
            agent.step_counter += 1
            if agent.step_counter % agent.learning_freq == 0:
                loss = agent.train()
                if loss is not None:
                    episode_loss += loss
                    loss_count += 1
            
            # 更新状态
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
        
        # 计算平均损失
        avg_loss = episode_loss / loss_count if loss_count > 0 else 0
        
        # 统计发车次数和时间
        up_departure_times = [i for i, a in enumerate(episode_actions) if a[0] == 1]
        down_departure_times = [i for i, a in enumerate(episode_actions) if a[1] == 1]
        
        # 应用双向平衡修正算法（论文要求）
        up_departure_times_balanced, down_departure_times_balanced = optimizer.balance_timetable(
            up_departure_times, down_departure_times, config.service_end
        )
        
        up_deps = len(up_departure_times_balanced)
        down_deps = len(down_departure_times_balanced)
        
        # 记录数据
        monitor.log_episode(episode, episode_reward, avg_loss, up_deps, down_deps, agent.epsilon)
        
        # 定期输出
        if (episode + 1) % 50 == 0:
            # 获取环境统计信息
            stats = env.get_statistics()
            
            print(f"\nEpisode {episode + 1}/{episodes}")
            avg_window = min(50, len(monitor.episode_rewards))
            print(f"  平均奖励(最近{avg_window}轮): {np.mean(monitor.episode_rewards[-avg_window:]):.2f}")
            print(f"  平均损失(最近{avg_window}轮): {np.mean(monitor.episode_losses[-avg_window:]):.4f}")
            print(f"  上行发车: {up_deps}, 下行发车: {down_deps}")
            print(f"  发车次数差: {abs(up_deps - down_deps)}")
            print(f"  总发车: {up_deps + down_deps}")
            
            # 显示乘客服务质量指标
            up_stats = stats['up_statistics']
            down_stats = stats['down_statistics']
            
            # 计算平均等待时间（分钟）
            up_avg_wait = up_stats['total_waiting_time'] / up_stats['total_passengers_served'] if up_stats['total_passengers_served'] > 0 else 0
            down_avg_wait = down_stats['total_waiting_time'] / down_stats['total_passengers_served'] if down_stats['total_passengers_served'] > 0 else 0
            
            print(f"  上行: 平均等待{up_avg_wait:.1f}分钟, 滞留乘客{up_stats['stranded_passengers']}人")
            print(f"  下行: 平均等待{down_avg_wait:.1f}分钟, 滞留乘客{down_stats['stranded_passengers']}人")
            
            print(f"  平均差异(最近{avg_window}轮): {np.mean(monitor.departure_differences[-avg_window:]):.2f}")
            print(f"  Epsilon: {agent.epsilon:.3f}")
        
        # 定期保存模型
        if (episode + 1) % save_interval == 0:
            model_path = monitor.save_dir / f'model_episode_{episode+1}.pth'
            agent.save_model(str(model_path))
            print(f"  模型已保存: {model_path}")
    
    print("\n" + "=" * 80)
    print("训练完成！")
    print("=" * 80)
    
    # 保存最终模型
    final_model_path = monitor.save_dir / 'model_final.pth'
    agent.save_model(str(final_model_path))
    print(f"\n最终模型已保存: {final_model_path}")
    
    # 保存训练指标
    monitor.save_metrics()
    
    # 绘制训练曲线
    monitor.plot_training_curves()
    
    # 获取最终统计
    final_stats = env.get_statistics()
    up_final = final_stats['up_statistics']
    down_final = final_stats['down_statistics']
    
    # 计算平均等待时间
    up_avg_wait_final = up_final['total_waiting_time'] / up_final['total_passengers_served'] if up_final['total_passengers_served'] > 0 else 0
    down_avg_wait_final = down_final['total_waiting_time'] / down_final['total_passengers_served'] if down_final['total_passengers_served'] > 0 else 0
    
    # 输出最终统计
    final_window = min(50, len(monitor.episode_rewards))
    print(f"\n最终统计:")
    print(f"  平均奖励(最后{final_window}轮): {np.mean(monitor.episode_rewards[-final_window:]):.2f}")
    print(f"  平均上行发车: {np.mean(monitor.up_departures[-final_window:]):.1f}")
    print(f"  平均下行发车: {np.mean(monitor.down_departures[-final_window:]):.1f}")
    print(f"  平均总发车: {np.mean([u + d for u, d in zip(monitor.up_departures[-final_window:], monitor.down_departures[-final_window:])]):.1f}")
    print(f"  发车次数平衡度(差异): {abs(np.mean(monitor.up_departures[-final_window:]) - np.mean(monitor.down_departures[-final_window:])):.2f}")
    print(f"  平均差异(最后{final_window}轮): {np.mean(monitor.departure_differences[-final_window:]):.2f}")
    print(f"  平衡约束违反次数: {len(monitor.balance_violations)} / {episodes}")
    
    print(f"\n服务质量指标:")
    print(f"  上行: 平均等待{up_avg_wait_final:.1f}分钟, 滞留乘客{up_final['stranded_passengers']}人")
    print(f"  下行: 平均等待{down_avg_wait_final:.1f}分钟, 滞留乘客{down_final['stranded_passengers']}人")
    
    # 目标达成情况
    avg_diff = np.mean(monitor.departure_differences[-final_window:])
    avg_total = np.mean([u + d for u, d in zip(monitor.up_departures[-final_window:], monitor.down_departures[-final_window:])])
    print(f"\n目标达成情况:")
    print(f"  ✓ 发车次数差异 < 1: {'是' if avg_diff < 1 else '否'} (当前: {avg_diff:.2f})")
    print(f"  ✓ 总发车次数 70-76: {'是' if 140 <= avg_total <= 152 else '否'} (当前: {avg_total:.1f}, 单向: {avg_total/2:.1f})")
    
    # 与论文对比
    print(f"\n与论文表2-3对比 (208线):")
    print(f"  {'指标':<20} {'论文(DRL-TSBC)':<15} {'我们的结果':<15}")
    print(f"  {'-'*50}")
    print(f"  {'上行发车次数':<20} {'73':<15} {f'{np.mean(monitor.up_departures[-final_window:]):.0f}':<15}")
    print(f"  {'下行发车次数':<20} {'73':<15} {f'{np.mean(monitor.down_departures[-final_window:]):.0f}':<15}")
    print(f"  {'上行平均等待(分钟)':<20} {'3.7':<15} {f'{up_avg_wait_final:.1f}':<15}")
    print(f"  {'下行平均等待(分钟)':<20} {'3.8':<15} {f'{down_avg_wait_final:.1f}':<15}")
    print(f"  {'上行滞留乘客':<20} {'0':<15} {up_final['stranded_passengers']:<15}")
    print(f"  {'下行滞留乘客':<20} {'0':<15} {down_final['stranded_passengers']:<15}")
    
    return agent, monitor


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='训练DRL-TSBC模型')
    parser.add_argument('--route', type=str, default='208', help='路线编号')
    parser.add_argument('--episodes', type=int, default=500, help='训练轮数 (默认500)')
    parser.add_argument('--save-interval', type=int, default=50, help='模型保存间隔')
    parser.add_argument('--device', type=str, default='auto', help='训练设备 (auto/cuda/cpu)')
    
    args = parser.parse_args()
    
    # 确定设备
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # 开始训练
    agent, monitor = train_drl_tsbc(
        route_id=args.route,
        episodes=args.episodes,
        save_interval=args.save_interval,
        device=device
    )
