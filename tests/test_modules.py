"""模块测试脚本"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_imports():
    """测试所有模块是否可以正常导入"""
    print("测试模块导入...")
    
    try:
        # 配置模块
        from utils.config import ConfigManager, RouteConfig, DQNConfig, ExperimentConfig
        print("  ✓ utils.config")
        
        # 数据加载模块
        from data.data_loader import PassengerDataLoader, TrafficDataLoader
        print("  ✓ data.data_loader")
        
        # 实体类
        from environment.entities import Passenger, Bus
        print("  ✓ environment.entities")
        
        # 环境模块
        from environment.bus_env import BusEnvironment
        print("  ✓ environment.bus_env")
        
        # DQN模块
        from models.dqn_network import DQN
        from models.replay_buffer import ReplayBuffer
        from models.dqn_agent import DQNAgent
        print("  ✓ models.dqn_network")
        print("  ✓ models.replay_buffer")
        print("  ✓ models.dqn_agent")
        
        # 推理模块
        from inference.schedule_generator import ScheduleGenerator
        from inference.evaluator import ScheduleEvaluator
        print("  ✓ inference.schedule_generator")
        print("  ✓ inference.evaluator")
        
        # 可视化模块
        from visualization.visualizer import Visualizer
        print("  ✓ visualization.visualizer")
        
        print("\n所有模块导入成功！")
        return True
        
    except Exception as e:
        print(f"\n模块导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config():
    """测试配置模块"""
    print("\n测试配置模块...")
    
    try:
        from utils.config import ConfigManager
        
        # 测试208线路配置
        config_208 = ConfigManager.get_route_208_config(0)
        assert config_208.route_id == 208
        assert config_208.direction == 0
        assert config_208.num_stations == 26
        print("  ✓ 208线路配置正确")
        
        # 测试211线路配置
        config_211 = ConfigManager.get_route_211_config(1)
        assert config_211.route_id == 211
        assert config_211.direction == 1
        assert config_211.num_stations == 11
        print("  ✓ 211线路配置正确")
        
        # 测试DQN配置
        dqn_config = ConfigManager.get_default_dqn_config()
        assert dqn_config.hidden_dim == 500
        assert dqn_config.num_layers == 12
        assert dqn_config.batch_size == 64
        print("  ✓ DQN配置正确")
        
        print("\n配置模块测试通过！")
        return True
        
    except Exception as e:
        print(f"\n配置模块测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_files():
    """测试数据文件是否存在"""
    print("\n测试数据文件...")
    
    data_files = [
        'test_data/208/passenger_dataframe_direction0.csv',
        'test_data/208/passenger_dataframe_direction1.csv',
        'test_data/208/traffic-0.csv',
        'test_data/208/traffic-1.csv',
    ]
    
    all_exist = True
    for filepath in data_files:
        if os.path.exists(filepath):
            print(f"  ✓ {filepath}")
        else:
            print(f"  ✗ {filepath} 不存在")
            all_exist = False
    
    if all_exist:
        print("\n所有数据文件存在！")
    else:
        print("\n警告：部分数据文件缺失")
    
    return all_exist


def test_directories():
    """测试目录结构"""
    print("\n测试目录结构...")
    
    directories = [
        'data',
        'environment',
        'models',
        'inference',
        'visualization',
        'utils',
        'experiments',
        'results/models',
        'results/logs',
        'results/figures',
        'results/tables',
        'configs'
    ]
    
    all_exist = True
    for directory in directories:
        if os.path.exists(directory):
            print(f"  ✓ {directory}/")
        else:
            print(f"  ✗ {directory}/ 不存在")
            all_exist = False
    
    if all_exist:
        print("\n所有目录存在！")
    else:
        print("\n警告：部分目录缺失")
    
    return all_exist


def test_dqn_network():
    """测试DQN网络"""
    print("\n测试DQN网络...")
    
    try:
        import torch
        from models.dqn_network import DQN
        
        # 创建网络
        dqn = DQN(state_dim=10, action_dim=4, hidden_dim=500, num_layers=12)
        print(f"  ✓ 网络创建成功")
        
        # 测试前向传播
        state = torch.randn(32, 10)
        q_values = dqn(state)
        assert q_values.shape == (32, 4)
        print(f"  ✓ 前向传播正确 (输入: {state.shape}, 输出: {q_values.shape})")
        
        # 测试参数数量
        total_params = sum(p.numel() for p in dqn.parameters())
        print(f"  ✓ 网络参数数量: {total_params:,}")
        
        print("\nDQN网络测试通过！")
        return True
        
    except Exception as e:
        print(f"\nDQN网络测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """运行所有测试"""
    print("="*70)
    print(" "*25 + "模块测试")
    print("="*70)
    
    results = []
    
    # 运行各项测试
    results.append(("模块导入", test_imports()))
    results.append(("配置模块", test_config()))
    results.append(("数据文件", test_data_files()))
    results.append(("目录结构", test_directories()))
    results.append(("DQN网络", test_dqn_network()))
    
    # 总结
    print("\n" + "="*70)
    print(" "*25 + "测试总结")
    print("="*70)
    
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{name:20s}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "="*70)
    if all_passed:
        print("所有测试通过！项目已准备就绪。")
    else:
        print("部分测试失败，请检查错误信息。")
    print("="*70)
    
    return all_passed


if __name__ == '__main__':
    run_all_tests()
