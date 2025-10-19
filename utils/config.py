"""配置管理模块"""
import json
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class RouteConfig:
    """线路配置"""
    route_id: int
    direction: int  # 0=上行, 1=下行
    num_stations: int
    start_time: int  # 首班车时间(分钟)
    end_time: int  # 末班车时间(分钟)
    max_capacity: int = 100  # 最大载客量
    seats: int = 40  # 座位数
    t_min: int = 5  # 最小发车间隔(分钟)
    t_max: int = 20  # 最大发车间隔(分钟)
    omega: float = 1/1000  # 等待时间惩罚权重
    
    # 奖励函数参数
    beta: float = 0.2  # 滞留乘客惩罚权重
    zeta: float = 0.002  # 发车平衡权重
    mu: float = 5000.0  # 等待时间归一化参数
    delta: float = 200.0  # 发车次数归一化参数
    alpha: float = 1.5  # 站立系数


@dataclass
class DQNConfig:
    """DQN配置"""
    state_dim: int = 10
    action_dim: int = 4
    hidden_dim: int = 500
    num_layers: int = 12
    learning_rate: float = 0.001
    gamma: float = 0.4  # 折扣因子
    epsilon: float = 0.1  # 探索率
    batch_size: int = 64
    replay_buffer_size: int = 3000
    learning_freq: int = 5  # 学习频率
    target_update_freq: int = 100  # 目标网络更新频率
    num_episodes: int = 50


@dataclass
class ExperimentConfig:
    """实验配置"""
    route_config: RouteConfig
    dqn_config: DQNConfig
    seed: int = 42
    device: str = 'cuda'
    save_dir: str = 'results/models'
    log_dir: str = 'results/logs'
    figure_dir: str = 'results/figures'
    table_dir: str = 'results/tables'


class ConfigManager:
    """配置管理器"""
    
    @staticmethod
    def get_route_208_config(direction: int) -> RouteConfig:
        """
        获取208线路配置
        
        Args:
            direction: 0=上行, 1=下行
            
        Returns:
            RouteConfig对象
        """
        return RouteConfig(
            route_id=208,
            direction=direction,
            num_stations=26 if direction == 0 else 24,
            start_time=6*60,  # 6:00
            end_time=21*60,  # 21:00
            max_capacity=100,
            seats=40,
            t_min=5,
            t_max=20,
            omega=1/1000
        )
    
    @staticmethod
    def get_route_211_config(direction: int) -> RouteConfig:
        """
        获取211线路配置
        
        Args:
            direction: 0=上行, 1=下行
            
        Returns:
            RouteConfig对象
        """
        return RouteConfig(
            route_id=211,
            direction=direction,
            num_stations=17 if direction == 0 else 11,
            start_time=6*60,  # 6:00
            end_time=22*60,  # 22:00
            max_capacity=100,
            seats=40,
            t_min=5,
            t_max=20,
            omega=1/900
        )
    
    @staticmethod
    def get_default_dqn_config() -> DQNConfig:
        """获取默认DQN配置"""
        return DQNConfig()
    
    @staticmethod
    def save_config(config: ExperimentConfig, path: str):
        """
        保存配置到JSON文件
        
        Args:
            config: 实验配置对象
            path: 保存路径
        """
        config_dict = {
            'route_config': asdict(config.route_config),
            'dqn_config': asdict(config.dqn_config),
            'seed': config.seed,
            'device': config.device,
            'save_dir': config.save_dir,
            'log_dir': config.log_dir,
            'figure_dir': config.figure_dir,
            'table_dir': config.table_dir
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def load_config(path: str) -> ExperimentConfig:
        """
        从JSON文件加载配置
        
        Args:
            path: 配置文件路径
            
        Returns:
            ExperimentConfig对象
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        route_config = RouteConfig(**data['route_config'])
        dqn_config = DQNConfig(**data['dqn_config'])
        
        return ExperimentConfig(
            route_config=route_config,
            dqn_config=dqn_config,
            seed=data.get('seed', 42),
            device=data.get('device', 'cuda'),
            save_dir=data.get('save_dir', 'results/models'),
            log_dir=data.get('log_dir', 'results/logs'),
            figure_dir=data.get('figure_dir', 'results/figures'),
            table_dir=data.get('table_dir', 'results/tables')
        )
    
    @staticmethod
    def create_default_configs():
        """创建默认配置文件"""
        import os
        os.makedirs('configs', exist_ok=True)
        
        # 208线路上行
        config_208_0 = ExperimentConfig(
            route_config=ConfigManager.get_route_208_config(0),
            dqn_config=ConfigManager.get_default_dqn_config()
        )
        ConfigManager.save_config(config_208_0, 'configs/route_208_dir_0.json')
        
        # 208线路下行
        config_208_1 = ExperimentConfig(
            route_config=ConfigManager.get_route_208_config(1),
            dqn_config=ConfigManager.get_default_dqn_config()
        )
        ConfigManager.save_config(config_208_1, 'configs/route_208_dir_1.json')
        
        # 211线路上行
        config_211_0 = ExperimentConfig(
            route_config=ConfigManager.get_route_211_config(0),
            dqn_config=ConfigManager.get_default_dqn_config()
        )
        ConfigManager.save_config(config_211_0, 'configs/route_211_dir_0.json')
        
        # 211线路下行
        config_211_1 = ExperimentConfig(
            route_config=ConfigManager.get_route_211_config(1),
            dqn_config=ConfigManager.get_default_dqn_config()
        )
        ConfigManager.save_config(config_211_1, 'configs/route_211_dir_1.json')


if __name__ == '__main__':
    # 创建默认配置文件
    ConfigManager.create_default_configs()
    print("配置文件创建完成！")
