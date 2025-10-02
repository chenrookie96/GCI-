# src/environment/bus_simulator.py
"""
公交系统仿真环境
模拟双向公交线路的运营环境
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import random

@dataclass
class BusTrip:
    """公交行程"""
    departure_time: int  # 发车时间(分钟)
    arrival_time: int    # 到达时间(分钟)
    direction: str       # 方向 ('up' or 'down')
    passenger_count: int # 乘客数量
    capacity: int        # 车辆容量

@dataclass
class PassengerFlow:
    """乘客流量"""
    time: int           # 时间(分钟)
    station_id: int     # 站点ID
    boarding: int       # 上车人数
    alighting: int      # 下车人数

class BusSystemEnvironment:
    """公交系统仿真环境"""
    
    def __init__(self, 
                 service_start: int = 360,    # 服务开始时间 (6:00 AM = 360分钟)
                 service_end: int = 1320,     # 服务结束时间 (10:00 PM = 1320分钟)
                 num_stations: int = 37,      # 站点数量
                 bus_capacity: int = 47,      # 公交车容量
                 avg_travel_time: int = 35):  # 平均行程时间(分钟)
        
        self.service_start = service_start
        self.service_end = service_end
        self.num_stations = num_stations
        self.bus_capacity = bus_capacity
        self.avg_travel_time = avg_travel_time
        
        # 当前状态
        self.current_time = service_start
        self.buses_in_service = []
        self.passenger_flows = {}
        self.waiting_passengers = {}
        self.stranded_passengers = 0
        
        # 统计信息
        self.total_departures = 0
        self.total_waiting_time = 0
        self.total_passengers_served = 0
        
        # 初始化乘客流量数据
        self._initialize_passenger_flow()
        
    def _initialize_passenger_flow(self):
        """初始化乘客流量数据"""
        # 模拟一天的乘客流量模式
        for minute in range(self.service_start, self.service_end + 1):
            hour = minute // 60
            
            # 根据时间段设置不同的乘客流量强度
            if 7 <= hour <= 9 or 17 <= hour <= 19:  # 高峰期
                base_flow = 50
            elif 10 <= hour <= 16:  # 平峰期
                base_flow = 20
            else:  # 低峰期
                base_flow = 10
                
            # 为每个站点生成乘客流量
            station_flows = []
            for station in range(self.num_stations):
                # 添加随机波动
                flow_variation = random.gauss(1.0, 0.3)
                station_flow = max(0, int(base_flow * flow_variation))
                
                station_flows.append(PassengerFlow(
                    time=minute,
                    station_id=station,
                    boarding=station_flow,
                    alighting=random.randint(0, station_flow)
                ))
                
            self.passenger_flows[minute] = station_flows
            
    def reset(self) -> Dict[str, Any]:
        """重置环境"""
        self.current_time = self.service_start
        self.buses_in_service = []
        self.waiting_passengers = {i: 0 for i in range(self.num_stations)}
        self.stranded_passengers = 0
        self.total_departures = 0
        self.total_waiting_time = 0
        self.total_passengers_served = 0
        
        return self._get_current_state()
        
    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool]:
        """执行一步仿真"""
        # action: 0 = 不发车, 1 = 发车
        
        if action == 1:  # 发车
            self._dispatch_bus()
            
        # 更新时间
        self.current_time += 1
        
        # 更新乘客流量
        self._update_passenger_flow()
        
        # 更新在途公交车状态
        self._update_buses()
        
        # 计算奖励
        reward = self._calculate_reward(action)
        
        # 检查是否结束
        done = self.current_time >= self.service_end
        
        return self._get_current_state(), reward, done
        
    def _dispatch_bus(self):
        """发车"""
        new_bus = BusTrip(
            departure_time=self.current_time,
            arrival_time=self.current_time + self.avg_travel_time,
            direction='up',  # 简化为单向
            passenger_count=0,
            capacity=self.bus_capacity
        )
        
        # 乘客上车
        passengers_to_board = min(
            self.waiting_passengers.get(0, 0),  # 起点站等待乘客
            self.bus_capacity
        )
        
        new_bus.passenger_count = passengers_to_board
        self.waiting_passengers[0] = max(0, self.waiting_passengers.get(0, 0) - passengers_to_board)
        
        # 计算滞留乘客
        self.stranded_passengers += max(0, self.waiting_passengers.get(0, 0) - passengers_to_board)
        
        self.buses_in_service.append(new_bus)
        self.total_departures += 1
        self.total_passengers_served += passengers_to_board
        
    def _update_passenger_flow(self):
        """更新乘客流量"""
        if self.current_time in self.passenger_flows:
            flows = self.passenger_flows[self.current_time]
            for flow in flows:
                station_id = flow.station_id
                if station_id not in self.waiting_passengers:
                    self.waiting_passengers[station_id] = 0
                self.waiting_passengers[station_id] += flow.boarding
                
    def _update_buses(self):
        """更新在途公交车状态"""
        buses_to_remove = []
        
        for i, bus in enumerate(self.buses_in_service):
            if self.current_time >= bus.arrival_time:
                # 公交车到达终点
                buses_to_remove.append(i)
                
        # 移除已到达的公交车
        for i in reversed(buses_to_remove):
            self.buses_in_service.pop(i)
            
    def _calculate_reward(self, action: int) -> float:
        """计算奖励（这里返回0，实际奖励在智能体中计算）"""
        return 0.0
        
    def _get_current_state(self) -> Dict[str, Any]:
        """获取当前状态"""
        # 计算状态特征
        hour = self.current_time // 60
        minute = self.current_time % 60
        
        # 计算最大载客率
        max_load_factor = 0.0
        if self.buses_in_service:
            max_passengers = max(bus.passenger_count for bus in self.buses_in_service)
            max_load_factor = max_passengers / self.bus_capacity
            
        # 计算标准化等待时间
        total_waiting = sum(self.waiting_passengers.values())
        normalized_waiting_time = min(total_waiting / 100.0, 1.0)  # 标准化到[0,1]
        
        # 计算运力利用率
        if self.total_departures > 0:
            capacity_utilization = self.total_passengers_served / (self.total_departures * self.bus_capacity)
        else:
            capacity_utilization = 0.0
            
        return {
            'time_hour': hour,
            'time_minute': minute,
            'max_load_factor': max_load_factor,
            'normalized_waiting_time': normalized_waiting_time,
            'capacity_utilization': capacity_utilization,
            'stranded_passengers': self.stranded_passengers,
            'waiting_passengers': total_waiting,
            'buses_in_service': len(self.buses_in_service),
            'current_time': self.current_time
        }
        
    def is_done(self) -> bool:
        """检查是否结束"""
        return self.current_time >= self.service_end
        
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'total_departures': self.total_departures,
            'total_passengers_served': self.total_passengers_served,
            'total_waiting_time': self.total_waiting_time,
            'stranded_passengers': self.stranded_passengers,
            'average_waiting_time': self.total_waiting_time / max(1, self.total_passengers_served),
            'service_utilization': self.total_passengers_served / max(1, self.total_departures * self.bus_capacity)
        }

class PassengerFlowGenerator:
    """乘客流量生成器"""
    
    @staticmethod
    def generate_realistic_flow(num_days: int = 1, 
                              stations: int = 37,
                              service_hours: Tuple[int, int] = (6, 22)) -> Dict[int, List[PassengerFlow]]:
        """生成真实的乘客流量数据"""
        flows = {}
        
        for day in range(num_days):
            day_offset = day * 24 * 60
            
            for hour in range(service_hours[0], service_hours[1] + 1):
                for minute in range(60):
                    time_key = day_offset + hour * 60 + minute
                    
                    # 根据时间段和站点特征生成流量
                    station_flows = []
                    
                    for station in range(stations):
                        # 不同站点的流量特征
                        if station in [0, stations-1]:  # 起终点站
                            base_flow = 30
                        elif station in range(5, 10):  # 商业区
                            base_flow = 25
                        elif station in range(15, 20):  # 住宅区
                            base_flow = 20
                        else:  # 普通站点
                            base_flow = 15
                            
                        # 时间段调整
                        if hour in [7, 8, 17, 18]:  # 高峰期
                            time_factor = 2.0
                        elif hour in [9, 10, 16]:  # 次高峰
                            time_factor = 1.5
                        else:  # 平峰期
                            time_factor = 1.0
                            
                        # 添加随机性
                        flow_value = int(base_flow * time_factor * random.uniform(0.5, 1.5))
                        
                        station_flows.append(PassengerFlow(
                            time=time_key,
                            station_id=station,
                            boarding=max(0, flow_value),
                            alighting=max(0, int(flow_value * 0.8))
                        ))
                        
                    flows[time_key] = station_flows
                    
        return flows
