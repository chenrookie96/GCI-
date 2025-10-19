"""数据加载模块"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import os
from environment.entities import Passenger


class PassengerDataLoader:
    """乘客数据加载器"""
    
    def __init__(self, route_id: int, direction: int, data_dir: str = 'test_data'):
        """
        初始化乘客数据加载器
        
        Args:
            route_id: 线路编号 (208, 211, 683)
            direction: 方向 (0=上行, 1=下行)
            data_dir: 数据目录
        """
        self.route_id = route_id
        self.direction = direction
        self.data_dir = data_dir
        self.passenger_data = None
        self.passengers_by_time = {}  # 按时间索引的乘客字典
        
    def load_passenger_data(self) -> pd.DataFrame:
        """
        加载乘客数据
        
        Returns:
            包含乘客信息的DataFrame
        """
        file_path = os.path.join(
            self.data_dir, 
            str(self.route_id), 
            f'passenger_dataframe_direction{self.direction}.csv'
        )
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"乘客数据文件不存在: {file_path}")
        
        # 读取CSV文件
        self.passenger_data = pd.read_csv(file_path)
        
        # 验证必需的列
        required_columns = ['Label', 'Boarding time', 'Boarding station', 
                          'Alighting station', 'Arrival time']
        for col in required_columns:
            if col not in self.passenger_data.columns:
                raise ValueError(f"缺少必需的列: {col}")
        
        # 按到达时间组织乘客数据
        self._organize_passengers_by_time()
        
        return self.passenger_data
    
    def _organize_passengers_by_time(self):
        """按到达时间组织乘客数据"""
        self.passengers_by_time = {}
        
        for _, row in self.passenger_data.iterrows():
            arrival_time = int(row['Arrival time'])
            
            if arrival_time not in self.passengers_by_time:
                self.passengers_by_time[arrival_time] = []
            
            passenger = Passenger(
                label=str(row['Label']),
                arrival_time=arrival_time,
                boarding_time=-1,  # 初始未上车
                boarding_station=int(row['Boarding station']),
                alighting_station=int(row['Alighting station'])
            )
            
            self.passengers_by_time[arrival_time].append(passenger)
    
    def get_passengers_at_time(self, time_minute: int) -> List[Passenger]:
        """
        获取指定时间到达的乘客列表
        
        Args:
            time_minute: 时间(分钟)
            
        Returns:
            乘客列表
        """
        return self.passengers_by_time.get(time_minute, [])
    
    def get_all_passengers(self) -> List[Passenger]:
        """获取所有乘客"""
        all_passengers = []
        for passengers in self.passengers_by_time.values():
            all_passengers.extend(passengers)
        return all_passengers
    
    def get_passenger_count(self) -> int:
        """获取乘客总数"""
        return len(self.passenger_data) if self.passenger_data is not None else 0


class TrafficDataLoader:
    """交通数据加载器"""
    
    def __init__(self, route_id: int, direction: int, data_dir: str = 'test_data'):
        """
        初始化交通数据加载器
        
        Args:
            route_id: 线路编号
            direction: 方向 (0=上行, 1=下行)
            data_dir: 数据目录
        """
        self.route_id = route_id
        self.direction = direction
        self.data_dir = data_dir
        self.traffic_data = None
        self.travel_time_matrix = None
        
    def load_traffic_data(self) -> pd.DataFrame:
        """
        加载交通数据
        
        Returns:
            包含站间行驶时间的DataFrame
        """
        file_path = os.path.join(
            self.data_dir,
            str(self.route_id),
            f'traffic-{self.direction}.csv'
        )
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"交通数据文件不存在: {file_path}")
        
        # 读取CSV文件
        self.traffic_data = pd.read_csv(file_path)
        
        # 提取站间行驶时间矩阵
        self._extract_travel_time_matrix()
        
        return self.traffic_data
    
    def _extract_travel_time_matrix(self):
        """提取站间行驶时间矩阵"""
        # 获取站点列（s0, s1, s2, ...）
        station_cols = [col for col in self.traffic_data.columns if col.startswith('s')]
        
        # 创建时间段到行驶时间的映射
        self.travel_time_matrix = {}
        
        for _, row in self.traffic_data.iterrows():
            start_time = int(row['start_m'])
            end_time = int(row['finish_m'])
            
            # 提取该时间段的站间行驶时间
            travel_times = [int(row[col]) for col in station_cols]
            
            # 为该时间段的每一分钟都存储行驶时间
            for t in range(start_time, end_time + 1):
                self.travel_time_matrix[t] = travel_times
    
    def get_travel_time(self, from_station: int, to_station: int, 
                       current_time: int) -> int:
        """
        获取站间行驶时间
        
        Args:
            from_station: 起始站点
            to_station: 目标站点
            current_time: 当前时间(分钟)
            
        Returns:
            行驶时间(分钟)
        """
        if current_time not in self.travel_time_matrix:
            # 如果没有该时间的数据，使用最近的时间段
            available_times = sorted(self.travel_time_matrix.keys())
            if not available_times:
                return 2  # 默认值
            
            # 找到最近的时间
            closest_time = min(available_times, key=lambda t: abs(t - current_time))
            current_time = closest_time
        
        travel_times = self.travel_time_matrix[current_time]
        
        # 计算从from_station到to_station的总行驶时间
        total_time = 0
        for station in range(from_station, to_station):
            if station < len(travel_times):
                total_time += travel_times[station]
            else:
                total_time += 2  # 默认值
        
        return max(1, total_time)  # 至少1分钟
    
    def get_next_station_time(self, from_station: int, current_time: int) -> int:
        """
        获取到下一站的行驶时间
        
        Args:
            from_station: 当前站点
            current_time: 当前时间(分钟)
            
        Returns:
            到下一站的行驶时间(分钟)
        """
        return self.get_travel_time(from_station, from_station + 1, current_time)


if __name__ == '__main__':
    # 测试数据加载
    print("测试乘客数据加载...")
    passenger_loader = PassengerDataLoader(route_id=208, direction=0)
    passenger_data = passenger_loader.load_passenger_data()
    print(f"加载了 {passenger_loader.get_passenger_count()} 条乘客数据")
    print(f"数据列: {list(passenger_data.columns)}")
    print(f"前5条数据:\n{passenger_data.head()}")
    
    print("\n测试交通数据加载...")
    traffic_loader = TrafficDataLoader(route_id=208, direction=0)
    traffic_data = traffic_loader.load_traffic_data()
    print(f"加载了 {len(traffic_data)} 条交通数据")
    print(f"数据列: {list(traffic_data.columns)}")
    
    # 测试获取行驶时间
    travel_time = traffic_loader.get_travel_time(0, 1, 360)  # 6:00
    print(f"\n站点0到站点1的行驶时间: {travel_time}分钟")
