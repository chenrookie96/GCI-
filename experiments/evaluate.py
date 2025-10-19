"""评估脚本"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import json

from utils.config import ConfigManager
from data.data_loader import PassengerDataLoader, TrafficDataLoader
from environment.bus_env import BusEnvironment
from models.dqn_agent import DQNAgent
from inference.schedule_generator import ScheduleGenerator
from inference.evaluator import ScheduleEvaluator
from visualization.visualizer import Visualizer


def evaluate(model_path: str, route_id: int, direction: int,
            save_schedule: bool = True):
    """
    评估训练好的模型
    
    Args:
        model_path: 模型路径
        route_id: 线路编号
        direction: 方向
        save_schedule: 是否保存时刻表
    """
    print("="*60)
    print("DRL-TSBC 模型评估")
    print("="*60)
    
    # 加载配置
    print("\n加载配置...")
    if route_id == 208:
        route_config = ConfigManager.get_route_208_config(direction)
    else:
        route_config = ConfigManager.get_route_211_config(direction)
    
    dqn_config = ConfigManager.get_default_dqn_config()
    
    print(f"  线路: {route_id}")
    print(f"  方向: {'上行' if direction == 0 else '下行'}")
    
    # 加载数据
    print("\n加载数据...")
    passenger_loader = PassengerDataLoader(route_id=route_id, direction=direction)
    passenger_loader.load_passenger_data()
    
    traffic_loader = TrafficDataLoader(route_id=route_id, direction=direction)
    traffic_loader.load_traffic_data()
    
    # 创建环境
    print("\n创建环境...")
    env = BusEnvironment(
        config=route_config,
        passenger_loader=passenger_loader,
        traffic_loader=traffic_loader
    )
    
    # 创建智能体并加载模型
    print("\n加载模型...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = DQNAgent(config=dqn_config, device=device)
    agent.load(model_path)
    
    # 生成时刻表
    print("\n生成时刻表...")
    generator = ScheduleGenerator(agent=agent, env=env)
    schedule = generator.generate_schedule()
    
    # 格式化并打印时刻表
    formatted_schedule = generator.format_schedule(schedule)
    print("\n上行时刻表:")
    print(", ".join(formatted_schedule['up'][:10]) + " ...")
    print(f"共 {len(schedule['up'])} 班次")
    
    print("\n下行时刻表:")
    print(", ".join(formatted_schedule['down'][:10]) + " ...")
    print(f"共 {len(schedule['down'])} 班次")
    
    # 保存时刻表
    if save_schedule:
        schedule_path = model_path.replace('.pth', '_schedule.json')
        generator.save_schedule(schedule, schedule_path)
    
    # 评估时刻表
    print("\n评估时刻表性能...")
    evaluator = ScheduleEvaluator(env=env)
    
    # 重新创建环境以进行评估
    env_eval = BusEnvironment(
        config=route_config,
        passenger_loader=passenger_loader,
        traffic_loader=traffic_loader
    )
    evaluator_eval = ScheduleEvaluator(env=env_eval)
    
    results = evaluator_eval.evaluate_schedule(schedule)
    
    # 打印评估结果
    print("\n评估结果:")
    print("-"*60)
    print(f"  上行发车次数: {results['dispatch_count_up']}")
    print(f"  下行发车次数: {results['dispatch_count_down']}")
    print(f"  总发车次数: {results['total_dispatch']}")
    print(f"  平均等待时间: {results['avg_waiting_time']:.2f} 分钟")
    print(f"  滞留乘客数: {results['stranded_passengers']}")
    print(f"  服务乘客数: {results['total_passengers_served']}")
    print("-"*60)
    
    # 保存评估结果
    results_path = model_path.replace('.pth', '_evaluation.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n评估结果已保存到: {results_path}")
    
    # 生成可视化
    print("\n生成可视化...")
    vis = Visualizer()
    
    # 时刻表热力图
    vis.plot_schedule_heatmap(
        schedule=schedule,
        filename=f'schedule_heatmap_route_{route_id}_dir_{direction}.png'
    )
    
    print("\n评估完成！")
    print("="*60)
    
    return schedule, results


def main():
    parser = argparse.ArgumentParser(description='评估DRL-TSBC模型')
    parser.add_argument('--model', type=str, required=True,
                       help='模型文件路径')
    parser.add_argument('--route', type=int, default=208, choices=[208, 211],
                       help='线路编号')
    parser.add_argument('--direction', type=int, default=0, choices=[0, 1],
                       help='方向 (0=上行, 1=下行)')
    parser.add_argument('--no-save', action='store_true',
                       help='不保存时刻表')
    
    args = parser.parse_args()
    
    evaluate(
        model_path=args.model,
        route_id=args.route,
        direction=args.direction,
        save_schedule=not args.no_save
    )


if __name__ == '__main__':
    main()
