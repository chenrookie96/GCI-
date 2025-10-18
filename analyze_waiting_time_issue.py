"""
分析等待时间过长的原因
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config.bus_config import get_route_config
from src.data_processing.passenger_data_loader import PassengerDataLoader
from src.environment.station_level_simulator import StationLevelBusEnvironment
import numpy as np

config = get_route_config('208')
data_loader = PassengerDataLoader('test_data')
route_data = data_loader.load_route_data('208')

env = StationLevelBusEnvironment(
    service_start=config.service_start,
    service_end=config.service_end,
    num_stations=config.num_stations_up,
    bus_capacity=config.capacity,
    avg_travel_time=config.avg_travel_time,
    enable_logging=False
)

env.load_passenger_data(route_data, filter_out_of_service=True, convert_station_ids=True)

print("=" * 80)
print("分析等待时间问题")
print("=" * 80)

# 统计乘客分布
print(f"\n乘客分布:")
print(f"  上行总乘客: {sum(len(arrivals.get('up', [])) for arrivals in env.passenger_arrivals.values())}")
print(f"  下行总乘客: {sum(len(arrivals.get('down', [])) for arrivals in env.passenger_arrivals.values())}")

# 分析乘客到达的时间分布
up_arrival_times = []
down_arrival_times = []

for time_step, arrivals in env.passenger_arrivals.items():
    for p in arrivals.get('up', []):
        up_arrival_times.append(p.arrival_time)
    for p in arrivals.get('down', []):
        down_arrival_times.append(p.arrival_time)

print(f"\n上行乘客到达时间分布:")
print(f"  平均到达时间: {np.mean(up_arrival_times):.0f} ({int(np.mean(up_arrival_times))//60:02d}:{int(np.mean(up_arrival_times))%60:02d})")
print(f"  中位数: {np.median(up_arrival_times):.0f} ({int(np.median(up_arrival_times))//60:02d}:{int(np.median(up_arrival_times))%60:02d})")
print(f"  标准差: {np.std(up_arrival_times):.0f}分钟")

print(f"\n下行乘客到达时间分布:")
print(f"  平均到达时间: {np.mean(down_arrival_times):.0f} ({int(np.mean(down_arrival_times))//60:02d}:{int(np.mean(down_arrival_times))%60:02d})")
print(f"  中位数: {np.median(down_arrival_times):.0f} ({int(np.median(down_arrival_times))//60:02d}:{int(np.median(down_arrival_times))%60:02d})")
print(f"  标准差: {np.std(down_arrival_times):.0f}分钟")

# 分析各站点的乘客分布
print(f"\n上行各站点乘客数（前10个站点）:")
up_station_counts = {}
for arrivals in env.passenger_arrivals.values():
    for p in arrivals.get('up', []):
        up_station_counts[p.boarding_station] = up_station_counts.get(p.boarding_station, 0) + 1

for station_id in sorted(up_station_counts.keys())[:10]:
    print(f"  站点{station_id}: {up_station_counts[station_id]}人")

print(f"\n下行各站点乘客数（前10个站点）:")
down_station_counts = {}
for arrivals in env.passenger_arrivals.values():
    for p in arrivals.get('down', []):
        down_station_counts[p.boarding_station] = down_station_counts.get(p.boarding_station, 0) + 1

for station_id in sorted(down_station_counts.keys())[:10]:
    print(f"  站点{station_id}: {down_station_counts[station_id]}人")

# 理论最优等待时间
print(f"\n理论分析:")
service_duration = config.service_end - config.service_start
avg_interval = service_duration / 73  # 73次发车
print(f"  服务时长: {service_duration}分钟")
print(f"  发车次数: 73次")
print(f"  平均发车间隔: {avg_interval:.1f}分钟")
print(f"  理论最小平均等待: {avg_interval/2:.1f}分钟（均匀到达假设）")

print(f"\n可能的问题:")
print(f"  1. 训练轮数不够（只训练了100轮）")
print(f"  2. 发车策略不够优化（DRL还在学习中）")
print(f"  3. 下行乘客分布可能更不均匀")
print(f"  4. 奖励函数权重可能需要调整")
