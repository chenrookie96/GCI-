"""
固定首末班车功能使用示例
演示如何使用新的固定发车机制
"""

from src.environment.station_level_simulator import StationLevelBusEnvironment
import numpy as np


def example_basic_usage():
    """基本使用示例"""
    print("=" * 60)
    print("示例1: 基本使用")
    print("=" * 60)
    
    # 创建环境
    env = StationLevelBusEnvironment(
        service_start=360,  # 6:00 AM
        service_end=420,    # 7:00 AM
        num_stations=10,
        bus_capacity=48,
        enable_logging=True  # 启用日志查看发车详情
    )
    
    # 重置环境
    state = env.reset()
    print(f"\n初始状态:")
    print(f"  当前时间: {state['current_time']}")
    print(f"  上行首班车已发: {state['up_first_bus_dispatched']}")
    print(f"  下行首班车已发: {state['down_first_bus_dispatched']}")
    
    # 执行第一步 - 首班车会自动发车
    print(f"\n执行第一步 (DRL动作: 不发车)...")
    state, reward, done = env.step((0, 0))
    print(f"  动作被覆盖: {state['action_overridden']}")
    print(f"  固定发车(上行): {state['fixed_dispatch_up']}")
    print(f"  固定发车(下行): {state['fixed_dispatch_down']}")
    
    # 获取统计信息
    stats = env.get_statistics()
    print(f"\n统计信息:")
    print(f"  上行 - 总发车: {stats['up_statistics']['total_departures']}, "
          f"固定: {stats['up_statistics']['fixed_departures']}, "
          f"DRL: {stats['up_statistics']['drl_departures']}")
    print(f"  下行 - 总发车: {stats['down_statistics']['total_departures']}, "
          f"固定: {stats['down_statistics']['fixed_departures']}, "
          f"DRL: {stats['down_statistics']['drl_departures']}")


def example_with_drl_agent():
    """与DRL智能体集成示例"""
    print("\n" + "=" * 60)
    print("示例2: 与DRL智能体集成")
    print("=" * 60)
    
    # 创建环境
    env = StationLevelBusEnvironment(
        service_start=360,
        service_end=380,  # 短时间用于演示
        num_stations=5,
        bus_capacity=48,
        enable_logging=False
    )
    
    # 简单的DRL策略：每3分钟发一次车
    class SimpleDRLAgent:
        def __init__(self):
            self.step_count = 0
        
        def select_action(self, state):
            self.step_count += 1
            # 每3分钟发一次车
            if self.step_count % 3 == 0:
                return (1, 1)
            return (0, 0)
    
    agent = SimpleDRLAgent()
    
    # 运行episode
    state = env.reset()
    done = False
    
    print(f"\n运行episode...")
    while not done:
        action = agent.select_action(state)
        state, reward, done = env.step(action)
        
        if state['action_overridden']:
            print(f"时间 {state['current_time']}: 固定发车覆盖了DRL动作")
    
    # 获取最终统计
    stats = env.get_statistics()
    print(f"\n最终统计:")
    print(f"  上行:")
    print(f"    总发车: {stats['up_statistics']['total_departures']}")
    print(f"    固定发车: {stats['up_statistics']['fixed_departures']}")
    print(f"    DRL发车: {stats['up_statistics']['drl_departures']}")
    print(f"    DRL控制比例: {stats['up_statistics']['drl_control_ratio']:.2%}")
    print(f"    首班车时间: {stats['up_statistics']['first_bus_time']}")
    print(f"    末班车时间: {stats['up_statistics']['last_bus_time']}")


def example_statistics_analysis():
    """统计信息分析示例"""
    print("\n" + "=" * 60)
    print("示例3: 统计信息分析")
    print("=" * 60)
    
    # 创建环境
    env = StationLevelBusEnvironment(
        service_start=360,
        service_end=400,
        num_stations=8,
        bus_capacity=48,
        enable_logging=False
    )
    
    # 运行多个episode
    num_episodes = 3
    all_stats = []
    
    for episode in range(num_episodes):
        env.reset()
        done = False
        
        # 随机策略
        while not done:
            action = (np.random.choice([0, 1]), np.random.choice([0, 1]))
            state, reward, done = env.step(action)
        
        stats = env.get_statistics()
        all_stats.append(stats)
    
    # 分析统计信息
    print(f"\n运行了 {num_episodes} 个episodes:")
    for i, stats in enumerate(all_stats):
        print(f"\nEpisode {i+1}:")
        print(f"  上行 DRL控制比例: {stats['up_statistics']['drl_control_ratio']:.2%}")
        print(f"  下行 DRL控制比例: {stats['down_statistics']['drl_control_ratio']:.2%}")
        print(f"  双向发车差异: {stats['bidirectional_constraints']['departure_count_difference']}")


def example_state_observation():
    """状态观察示例"""
    print("\n" + "=" * 60)
    print("示例4: 状态观察")
    print("=" * 60)
    
    # 创建环境
    env = StationLevelBusEnvironment(
        service_start=360,
        service_end=365,
        num_stations=5,
        bus_capacity=48,
        enable_logging=False
    )
    
    # 重置并观察状态
    state = env.reset()
    
    print(f"\n初始状态包含的固定发车相关字段:")
    print(f"  action_overridden: {state['action_overridden']}")
    print(f"  fixed_dispatch_up: {state['fixed_dispatch_up']}")
    print(f"  fixed_dispatch_down: {state['fixed_dispatch_down']}")
    print(f"  up_first_bus_dispatched: {state['up_first_bus_dispatched']}")
    print(f"  down_first_bus_dispatched: {state['down_first_bus_dispatched']}")
    
    # 执行一步
    state, _, _ = env.step((0, 0))
    
    print(f"\n首班车发车后的状态:")
    print(f"  action_overridden: {state['action_overridden']}")
    print(f"  fixed_dispatch_up: {state['fixed_dispatch_up']}")
    print(f"  fixed_dispatch_down: {state['fixed_dispatch_down']}")
    print(f"  up_first_bus_dispatched: {state['up_first_bus_dispatched']}")
    print(f"  down_first_bus_dispatched: {state['down_first_bus_dispatched']}")


if __name__ == '__main__':
    # 运行所有示例
    example_basic_usage()
    example_with_drl_agent()
    example_statistics_analysis()
    example_state_observation()
    
    print("\n" + "=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)
