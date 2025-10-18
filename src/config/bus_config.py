# src/config/bus_config.py
"""
公交系统配置
包含不同路线的运营参数
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class BusRouteConfig:
    """公交路线配置"""
    route_id: str
    seats: int  # 座位数
    alpha: float  # 站立系数
    capacity: int  # 总容量 = seats * alpha
    service_start: int  # 服务开始时间（分钟）
    service_end: int  # 服务结束时间（分钟）
    num_stations_up: int  # 上行站点数
    num_stations_down: int  # 下行站点数
    avg_travel_time: int  # 平均单程时间（分钟）
    tmin: int  # 最小发车间隔（分钟）
    tmax: int  # 最大发车间隔（分钟）


# 论文规范配置
PAPER_CONFIG = {
    'seats': 32,  # 座位数
    'alpha': 1.5,  # 站立系数
    'tmin': 3,  # 最小发车间隔
    'tmax': 12,  # 最大发车间隔（设为12分钟，接近论文73次发车的平均间隔12.3分钟）
}


# 路线配置
ROUTE_CONFIGS: Dict[str, BusRouteConfig] = {
    '208': BusRouteConfig(
        route_id='208',
        seats=PAPER_CONFIG['seats'],
        alpha=PAPER_CONFIG['alpha'],
        capacity=int(PAPER_CONFIG['seats'] * PAPER_CONFIG['alpha']),  # 32 * 1.5 = 48
        service_start=360,  # 6:00 AM
        service_end=1260,  # 21:00 PM (9:00 PM)
        num_stations_up=27,  # 根据数据统计
        num_stations_down=25,  # 根据数据统计
        avg_travel_time=1.2,  # 站点间平均时间（分钟）：假设单程30分钟 / 27站点 ≈ 1.1分钟
        tmin=PAPER_CONFIG['tmin'],
        tmax=PAPER_CONFIG['tmax']
    ),
    '211': BusRouteConfig(
        route_id='211',
        seats=PAPER_CONFIG['seats'],
        alpha=PAPER_CONFIG['alpha'],
        capacity=int(PAPER_CONFIG['seats'] * PAPER_CONFIG['alpha']),  # 48
        service_start=360,  # 6:00 AM
        service_end=1320,  # 22:00 PM (10:00 PM)
        num_stations_up=25,  # 需要根据实际数据调整
        num_stations_down=25,  # 需要根据实际数据调整
        avg_travel_time=30,  # 估算值
        tmin=PAPER_CONFIG['tmin'],
        tmax=PAPER_CONFIG['tmax']
    ),
    '683': BusRouteConfig(
        route_id='683',
        seats=PAPER_CONFIG['seats'],
        alpha=PAPER_CONFIG['alpha'],
        capacity=int(PAPER_CONFIG['seats'] * PAPER_CONFIG['alpha']),  # 48
        service_start=360,  # 6:00 AM
        service_end=1320,  # 22:00 PM
        num_stations_up=25,  # 需要根据实际数据调整
        num_stations_down=25,  # 需要根据实际数据调整
        avg_travel_time=30,  # 估算值
        tmin=PAPER_CONFIG['tmin'],
        tmax=PAPER_CONFIG['tmax']
    )
}


def get_route_config(route_id: str) -> BusRouteConfig:
    """
    获取路线配置
    
    Args:
        route_id: 路线编号
        
    Returns:
        路线配置对象
    """
    if route_id not in ROUTE_CONFIGS:
        raise ValueError(f"不支持的路线: {route_id}. 支持的路线: {list(ROUTE_CONFIGS.keys())}")
    
    return ROUTE_CONFIGS[route_id]


def print_route_config(route_id: str):
    """
    打印路线配置信息
    
    Args:
        route_id: 路线编号
    """
    config = get_route_config(route_id)
    
    print(f"\n路线 {route_id} 配置:")
    print(f"  座位数: {config.seats}")
    print(f"  站立系数α: {config.alpha}")
    print(f"  总容量: {config.capacity} 人")
    print(f"  服务时间: {config.service_start//60:02d}:{config.service_start%60:02d} - "
          f"{config.service_end//60:02d}:{config.service_end%60:02d}")
    print(f"  上行站点数: {config.num_stations_up}")
    print(f"  下行站点数: {config.num_stations_down}")
    print(f"  平均单程时间: {config.avg_travel_time} 分钟")
    print(f"  发车间隔约束: Tmin={config.tmin}分钟, Tmax={config.tmax}分钟")


def print_all_configs():
    """打印所有路线配置"""
    print("=" * 80)
    print("公交系统配置 (论文规范)")
    print("=" * 80)
    
    print(f"\n通用参数:")
    print(f"  座位数: {PAPER_CONFIG['seats']}")
    print(f"  站立系数α: {PAPER_CONFIG['alpha']}")
    print(f"  车辆容量: {int(PAPER_CONFIG['seats'] * PAPER_CONFIG['alpha'])} 人")
    print(f"  最小发车间隔Tmin: {PAPER_CONFIG['tmin']} 分钟")
    print(f"  最大发车间隔Tmax: {PAPER_CONFIG['tmax']} 分钟")
    
    for route_id in ROUTE_CONFIGS.keys():
        print_route_config(route_id)


if __name__ == "__main__":
    print_all_configs()
