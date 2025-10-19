"""DRL-TSBC 主入口文件"""
import sys
import os
import argparse

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiments.train import train
from experiments.evaluate import evaluate
from utils.config import ConfigManager, ExperimentConfig


def main():
    parser = argparse.ArgumentParser(
        description='DRL-TSBC: 基于深度强化学习的双向动态公交时刻表排班算法',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 训练208线路上行
  python main.py train --route 208 --direction 0 --episodes 50
  
  # 评估模型
  python main.py evaluate --model results/models/route_208_dir_0.pth --route 208 --direction 0
  
  # 使用配置文件训练
  python main.py train --config configs/route_208_dir_0.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # 训练命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--route', type=int, default=208, choices=[208, 211],
                             help='线路编号')
    train_parser.add_argument('--direction', type=int, default=0, choices=[0, 1],
                             help='方向 (0=上行, 1=下行)')
    train_parser.add_argument('--episodes', type=int, default=50,
                             help='训练episode数')
    train_parser.add_argument('--device', type=str, default='cuda',
                             help='设备 (cuda/cpu)')
    train_parser.add_argument('--seed', type=int, default=42,
                             help='随机种子')
    train_parser.add_argument('--config', type=str, default=None,
                             help='配置文件路径')
    
    # 评估命令
    eval_parser = subparsers.add_parser('evaluate', help='评估模型')
    eval_parser.add_argument('--model', type=str, required=True,
                            help='模型文件路径')
    eval_parser.add_argument('--route', type=int, default=208, choices=[208, 211],
                            help='线路编号')
    eval_parser.add_argument('--direction', type=int, default=0, choices=[0, 1],
                            help='方向 (0=上行, 1=下行)')
    eval_parser.add_argument('--no-save', action='store_true',
                            help='不保存时刻表')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        # 训练模式
        if args.config:
            config = ConfigManager.load_config(args.config)
        else:
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
        
        train(config)
        
    elif args.command == 'evaluate':
        # 评估模式
        evaluate(
            model_path=args.model,
            route_id=args.route,
            direction=args.direction,
            save_schedule=not args.no_save
        )
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
