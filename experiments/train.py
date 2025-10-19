"""训练脚本"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import numpy as np
from tqdm import tqdm

from utils.config import ConfigManager, ExperimentConfig
from data.data_loader import PassengerDataLoader, TrafficDataLoader
from environment.bus_env import BusEnvironment
from models.dqn_agent import DQNAgent
from visualization.visualizer import Visualizer


def set_seed(seed: int):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)


def train(config: ExperimentConfig, save_path: str = None):
    """
    训练DQN模型
    
    Args:
        config: 实验配置
        save_path: 模型保存路径
    """
    # 设置随机种子
    set_seed(config.seed)
    
    print("="*60)
    print("DRL-TSBC 训练开始")
    print("="*60)
    
    # 加载数据
    print("\n加载数据...")
    passenger_loader = PassengerDataLoader(
        route_id=config.route_config.route_id,
        direction=config.route_config.direction
    )
    passenger_loader.load_passenger_data()
    print(f"  乘客数据: {passenger_loader.get_passenger_count()} 条")
    
    traffic_loader = TrafficDataLoader(
        route_id=config.route_config.route_id,
        direction=config.route_config.direction
    )
    traffic_loader.load_traffic_data()
    print(f"  交通数据加载完成")
    
    # 创建环境
    print("\n创建仿真环境...")
    env = BusEnvironment(
        config=config.route_config,
        passenger_loader=passenger_loader,
        traffic_loader=traffic_loader
    )
    print(f"  线路: {config.route_config.route_id}")
    print(f"  方向: {'上行' if config.route_config.direction == 0 else '下行'}")
    print(f"  站点数: {config.route_config.num_stations}")
    print(f"  运营时间: {config.route_config.start_time//60}:00 - {config.route_config.end_time//60}:00")
    
    # 创建智能体
    print("\n创建DQN智能体...")
    agent = DQNAgent(config=config.dqn_config, device=config.device)
    print(f"  设备: {agent.device}")
    print(f"  网络参数: {sum(p.numel() for p in agent.policy_net.parameters()):,}")
    
    # 训练循环
    print(f"\n开始训练 ({config.dqn_config.num_episodes} episodes)...")
    print("-"*60)
    
    episode_rewards = []
    episode_dispatch_counts = []
    all_losses = []
    
    for episode in range(config.dqn_config.num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_loss = []
        done = False
        step = 0
        
        # 进度条
        pbar = tqdm(total=env.end_time - env.start_time,
                   desc=f"Episode {episode+1}/{config.dqn_config.num_episodes}",
                   leave=False)
        
        while not done:
            # 选择动作
            action_idx = agent.select_action(state)
            action = agent.action_index_to_tuple(action_idx)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 训练
            loss = agent.train_step(state, action, reward, next_state, done)
            if loss > 0:
                episode_loss.append(loss)
                all_losses.append(loss)
            
            episode_reward += reward
            state = next_state
            step += 1
            
            pbar.update(1)
        
        pbar.close()
        
        # 记录统计信息
        episode_rewards.append(episode_reward)
        episode_dispatch_counts.append({
            'up': env.dispatch_count_up,
            'down': env.dispatch_count_down
        })
        
        # 打印进度
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        print(f"Episode {episode+1:3d} | "
              f"Reward: {episode_reward:8.2f} | "
              f"Dispatch: Up={env.dispatch_count_up:3d}, Down={env.dispatch_count_down:3d} | "
              f"Loss: {avg_loss:.4f} | "
              f"Buffer: {len(agent.replay_buffer)}/{agent.replay_buffer.capacity}")
    
    print("-"*60)
    print("训练完成！")
    
    # 保存模型
    if save_path is None:
        os.makedirs(config.save_dir, exist_ok=True)
        save_path = os.path.join(
            config.save_dir,
            f"route_{config.route_config.route_id}_dir_{config.route_config.direction}.pth"
        )
    
    agent.save(save_path)
    
    # 保存训练数据
    import json
    training_data = {
        'episode_rewards': episode_rewards,
        'episode_dispatch_counts': episode_dispatch_counts,
        'config': {
            'route_id': config.route_config.route_id,
            'direction': config.route_config.direction,
            'num_episodes': config.dqn_config.num_episodes
        }
    }
    
    data_path = save_path.replace('.pth', '_training_data.json')
    with open(data_path, 'w') as f:
        json.dump(training_data, f, indent=2)
    print(f"训练数据已保存到: {data_path}")
    
    # 生成训练曲线
    print("\n生成训练曲线...")
    vis = Visualizer(output_dir=config.figure_dir)
    vis.plot_training_convergence(
        episode_rewards=episode_rewards,
        dispatch_counts=episode_dispatch_counts,
        title=f'DRL-TSBC 训练收敛曲线 (线路{config.route_config.route_id})',
        filename=f'convergence_route_{config.route_config.route_id}_dir_{config.route_config.direction}.png'
    )
    
    if all_losses:
        vis.plot_loss_curve(
            losses=all_losses,
            filename=f'loss_route_{config.route_config.route_id}_dir_{config.route_config.direction}.png'
        )
    
    print("\n训练流程全部完成！")
    print("="*60)
    
    return agent, episode_rewards, episode_dispatch_counts


def main():
    parser = argparse.ArgumentParser(description='训练DRL-TSBC模型')
    parser.add_argument('--route', type=int, default=208, choices=[208, 211],
                       help='线路编号')
    parser.add_argument('--direction', type=int, default=0, choices=[0, 1],
                       help='方向 (0=上行, 1=下行)')
    parser.add_argument('--episodes', type=int, default=50,
                       help='训练episode数')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备 (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径')
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config:
        config = ConfigManager.load_config(args.config)
    else:
        # 使用默认配置
        if args.route == 208:
            route_config = ConfigManager.get_route_208_config(args.direction)
        else:
            route_config = ConfigManager.get_route_211_config(args.direction)
        
        dqn_config = ConfigManager.get_default_dqn_config()
        dqn_config.num_episodes = args.episodes
        
        config = ExperimentConfig(
            route_config=route_config,
            dqn_config=dqn_config,
            seed=args.seed,
            device=args.device
        )
    
    # 训练
    train(config)


if __name__ == '__main__':
    main()
