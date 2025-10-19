"""批量训练所有线路"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from utils.config import ConfigManager, ExperimentConfig
from experiments.train import train
from experiments.evaluate import evaluate
import pandas as pd


def train_all_routes(episodes: int = 50, device: str = 'cuda'):
    """
    训练所有线路的所有方向
    
    Args:
        episodes: 训练episode数
        device: 设备
    """
    print("="*70)
    print(" "*20 + "批量训练所有线路")
    print("="*70)
    
    # 定义所有要训练的配置
    configs = [
        (208, 0, "208线路上行"),
        (208, 1, "208线路下行"),
        (211, 0, "211线路上行"),
        (211, 1, "211线路下行"),
    ]
    
    results_summary = []
    
    for route_id, direction, name in configs:
        print(f"\n\n{'='*70}")
        print(f"训练: {name}")
        print(f"{'='*70}")
        
        # 创建配置
        if route_id == 208:
            route_config = ConfigManager.get_route_208_config(direction)
        else:
            route_config = ConfigManager.get_route_211_config(direction)
        
        dqn_config = ConfigManager.get_default_dqn_config()
        dqn_config.num_episodes = episodes
        
        config = ExperimentConfig(
            route_config=route_config,
            dqn_config=dqn_config,
            seed=42,
            device=device
        )
        
        # 训练
        try:
            agent, episode_rewards, episode_dispatch_counts = train(config)
            
            # 评估
            model_path = f'results/models/route_{route_id}_dir_{direction}.pth'
            schedule, eval_results = evaluate(
                model_path=model_path,
                route_id=route_id,
                direction=direction,
                save_schedule=True
            )
            
            # 记录结果
            results_summary.append({
                '线路': route_id,
                '方向': '上行' if direction == 0 else '下行',
                '发车次数': eval_results['total_dispatch'],
                '平均等待时间(分钟)': eval_results['avg_waiting_time'],
                '滞留乘客数': eval_results['stranded_passengers'],
                '服务乘客数': eval_results['total_passengers_served'],
                '最终奖励': episode_rewards[-1] if episode_rewards else 0
            })
            
            print(f"\n{name} 训练和评估完成！")
            
        except Exception as e:
            print(f"\n{name} 训练失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 生成总结表格
    print(f"\n\n{'='*70}")
    print(" "*25 + "训练总结")
    print(f"{'='*70}\n")
    
    df = pd.DataFrame(results_summary)
    print(df.to_string(index=False))
    
    # 保存总结
    summary_path = 'results/tables/training_summary.csv'
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"\n训练总结已保存到: {summary_path}")
    
    print(f"\n{'='*70}")
    print("全部训练完成！")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description='批量训练所有线路')
    parser.add_argument('--episodes', type=int, default=50,
                       help='每个模型的训练episode数')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备 (cuda/cpu)')
    
    args = parser.parse_args()
    
    train_all_routes(episodes=args.episodes, device=args.device)


if __name__ == '__main__':
    main()
