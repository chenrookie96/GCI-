"""快速开始脚本 - 训练和评估208线路"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config import ConfigManager, ExperimentConfig
from experiments.train import train
from experiments.evaluate import evaluate


def quick_start():
    """快速开始：训练208线路上行和下行"""
    
    print("="*70)
    print(" "*20 + "DRL-TSBC 快速开始")
    print("="*70)
    print("\n本脚本将训练208线路的上行和下行模型")
    print("训练参数:")
    print("  - Episode数: 50")
    print("  - 学习率: 0.001")
    print("  - 批次大小: 64")
    print("  - 经验池大小: 3000")
    print("\n预计训练时间: 约30-60分钟 (取决于GPU性能)")
    print("="*70)
    
    input("\n按Enter键开始训练...")
    
    # 训练208线路上行
    print("\n\n" + "="*70)
    print("第1步: 训练208线路上行")
    print("="*70)
    
    route_config_up = ConfigManager.get_route_208_config(direction=0)
    dqn_config = ConfigManager.get_default_dqn_config()
    
    config_up = ExperimentConfig(
        route_config=route_config_up,
        dqn_config=dqn_config,
        seed=42,
        device='cuda'
    )
    
    agent_up, rewards_up, counts_up = train(config_up)
    model_path_up = 'results/models/route_208_dir_0.pth'
    
    print(f"\n上行模型训练完成！模型已保存到: {model_path_up}")
    
    # 评估上行模型
    print("\n\n" + "="*70)
    print("第2步: 评估208线路上行模型")
    print("="*70)
    
    schedule_up, results_up = evaluate(
        model_path=model_path_up,
        route_id=208,
        direction=0,
        save_schedule=True
    )
    
    # 训练208线路下行
    print("\n\n" + "="*70)
    print("第3步: 训练208线路下行")
    print("="*70)
    
    route_config_down = ConfigManager.get_route_208_config(direction=1)
    
    config_down = ExperimentConfig(
        route_config=route_config_down,
        dqn_config=dqn_config,
        seed=42,
        device='cuda'
    )
    
    agent_down, rewards_down, counts_down = train(config_down)
    model_path_down = 'results/models/route_208_dir_1.pth'
    
    print(f"\n下行模型训练完成！模型已保存到: {model_path_down}")
    
    # 评估下行模型
    print("\n\n" + "="*70)
    print("第4步: 评估208线路下行模型")
    print("="*70)
    
    schedule_down, results_down = evaluate(
        model_path=model_path_down,
        route_id=208,
        direction=1,
        save_schedule=True
    )
    
    # 总结
    print("\n\n" + "="*70)
    print(" "*25 + "训练完成总结")
    print("="*70)
    
    print("\n208线路上行:")
    print(f"  发车次数: {results_up['dispatch_count_up']}")
    print(f"  平均等待时间: {results_up['avg_waiting_time']:.2f} 分钟")
    print(f"  滞留乘客数: {results_up['stranded_passengers']}")
    
    print("\n208线路下行:")
    print(f"  发车次数: {results_down['dispatch_count_down']}")
    print(f"  平均等待时间: {results_down['avg_waiting_time']:.2f} 分钟")
    print(f"  滞留乘客数: {results_down['stranded_passengers']}")
    
    print("\n生成的文件:")
    print(f"  模型文件: {model_path_up}, {model_path_down}")
    print(f"  时刻表: results/models/route_208_dir_0_schedule.json")
    print(f"         results/models/route_208_dir_1_schedule.json")
    print(f"  图表: results/figures/")
    
    print("\n" + "="*70)
    print("全部完成！")
    print("="*70)


if __name__ == '__main__':
    try:
        quick_start()
    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
    except Exception as e:
        print(f"\n\n错误: {e}")
        import traceback
        traceback.print_exc()
