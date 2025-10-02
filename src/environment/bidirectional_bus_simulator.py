# src/environment/bidirectional_bus_simulator.py
"""
双向公交系统仿真环境
专门为DRL-TSBC算法设计的双向公交线路仿真环境
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import random

@dataclass
class BidirectionalBusTrip:
    """双向公交行程"""
    departure_time: int     # 发车时间(分钟)
    arrival_time: int       # 到达时间(分钟) 
    direction: str          # 方向 ('up' or 'down')
    passenger_count: int    # 乘客数量
    capacity: int           # 车辆容量
    trip_id: int           # 行程ID

@dataclass
class DirectionState:
    """单方向状态"""
    buses_in_service: List[BidirectionalBusTrip]
    waiting_passengers: int
    stranded_passengers: int
    total_departures: int
    total_passengers_served: int
    total_waiting_time: float
    last_departure_time: int

class BidirectionalBusEnvironment:
    """双向公交系统仿真环境"""
    
    def __init__(self, 
                 service_start: int = 360,      # 服务开始时间 (6:00 AM)
                 service_end: int = 1320,       # 服务结束时间 (10:00 PM)
                 num_stations: int = 20,        # 单方向站点数量
                 bus_capacity: int = 50,        # 公交车容量
                 avg_travel_time: int = 30):    # 平均单程时间(分钟)
        
        self.service_start = service_start
        self.service_end = service_end
        self.num_stations = num_stations
        self.bus_capacity = bus_capacity
        self.avg_travel_time = avg_travel_time
        
        # 当前状态
        self.current_time = service_start
        
        # 双向状态
        self.up_state = DirectionState(
            buses_in_service=[],
            waiting_passengers=0,
            stranded_passengers=0,
            total_departures=0,
            total_passengers_served=0,
            total_waiting_time=0.0,
            last_departure_time=-999
        )
        
        self.down_state = DirectionState(
            buses_in_service=[],
            waiting_passengers=0,
            stranded_passengers=0,
            total_departures=0,
            total_passengers_served=0,
            total_waiting_time=0.0,
            last_departure_time=-999
        )
        
        # 乘客流量数据
        self.passenger_flows = {}
        self.trip_counter = 0
        
        # 初始化乘客流量
        self._initialize_passenger_flow()
        
    def _initialize_passenger_flow(self):
        """初始化双向乘客流量数据"""
        for minute in range(self.service_start, self.service_end + 1):
            hour = minute // 60
            
            # 根据时间段和方向设置不同的乘客流量强度
            if 7 <= hour <= 9:  # 早高峰 - 上行流量大
                up_base_flow = 40
                down_base_flow = 15
            elif 17 <= hour <= 19:  # 晚高峰 - 下行流量大
                up_base_flow = 15
                down_base_flow = 40
            elif 10 <= hour <= 16:  # 平峰期
                up_base_flow = 20
                down_base_flow = 20
            else:  # 低峰期
                up_base_flow = 8
                down_base_flow = 8
                
            # 添加随机波动
            up_flow = max(0, int(up_base_flow * random.gauss(1.0, 0.3)))
            down_flow = max(0, int(down_base_flow * random.gauss(1.0, 0.3)))
            
            self.passenger_flows[minute] = {
                'up': up_flow,
                'down': down_flow
            }
            
    def reset(self) -> Dict[str, Any]:
        """重置环境"""
        self.current_time = self.service_start
        self.trip_counter = 0
        
        # 重置双向状态
        self.up_state = DirectionState(
            buses_in_service=[],
            waiting_passengers=0,
            stranded_passengers=0,
            total_departures=0,
            total_passengers_served=0,
            total_waiting_time=0.0,
            last_departure_time=-999
        )
        
        self.down_state = DirectionState(
            buses_in_service=[],
            waiting_passengers=0,
            stranded_passengers=0,
            total_departures=0,
            total_passengers_served=0,
            total_waiting_time=0.0,
            last_departure_time=-999
        )
        
        return self._get_current_state()
        
    def step(self, action: Tuple[int, int]) -> Tuple[Dict[str, Any], float, bool]:
        """
        执行一步仿真
        
        Args:
            action: (a_up, a_down) - 上行和下行的发车决策
            
        Returns:
            (next_state, reward, done)
        """
        a_up, a_down = action
        
        # 执行发车决策
        if a_up == 1:
            self._dispatch_bus('up')
            
        if a_down == 1:
            self._dispatch_bus('down')
            
        # 更新时间
        self.current_time += 1
        
        # 更新乘客流量
        self._update_passenger_flow()
        
        # 更新在途公交车状态
        self._update_buses()
        
        # 检查是否结束
        done = self.current_time >= self.service_end
        
        return self._get_current_state(), 0.0, done  # 奖励在智能体中计算
        
    def _dispatch_bus(self, direction: str):
        """发车"""
        self.trip_counter += 1
        
        # 获取对应方向的状态
        state = self.up_state if direction == 'up' else self.down_state
        
        # 创建新行程
        new_trip = BidirectionalBusTrip(
            departure_time=self.current_time,
            arrival_time=self.current_time + self.avg_travel_time,
            direction=direction,
            passenger_count=0,
            capacity=self.bus_capacity,
            trip_id=self.trip_counter
        )
        
        # 乘客上车
        passengers_to_board = min(state.waiting_passengers, self.bus_capacity)
        new_trip.passenger_count = passengers_to_board
        
        # 更新状态
        state.waiting_passengers = max(0, state.waiting_passengers - passengers_to_board)
        state.stranded_passengers += max(0, state.waiting_passengers - passengers_to_board)
        state.buses_in_service.append(new_trip)
        state.total_departures += 1
        state.total_passengers_served += passengers_to_board
        state.last_departure_time = self.current_time
        
        # 计算等待时间（简化计算）
        if passengers_to_board > 0:
            avg_waiting = (self.current_time - state.last_departure_time) / 2
            state.total_waiting_time += avg_waiting * passengers_to_board
            
    def _update_passenger_flow(self):
        """更新双向乘客流量"""
        if self.current_time in self.passenger_flows:
            flows = self.passenger_flows[self.current_time]
            self.up_state.waiting_passengers += flows['up']
            self.down_state.waiting_passengers += flows['down']
            
    def _update_buses(self):
        """更新在途公交车状态"""
        # 更新上行车辆
        buses_to_remove = []
        for i, bus in enumerate(self.up_state.buses_in_service):
            if self.current_time >= bus.arrival_time:
                buses_to_remove.append(i)
                
        for i in reversed(buses_to_remove):
            self.up_state.buses_in_service.pop(i)
            
        # 更新下行车辆
        buses_to_remove = []
        for i, bus in enumerate(self.down_state.buses_in_service):
            if self.current_time >= bus.arrival_time:
                buses_to_remove.append(i)
                
        for i in reversed(buses_to_remove):
            self.down_state.buses_in_service.pop(i)
            
    def _calculate_capacity_utilization(self, state: DirectionState) -> float:
        """计算运力利用率"""
        if state.total_departures == 0:
            return 0.0
        return state.total_passengers_served / (state.total_departures * self.bus_capacity)
        
    def _calculate_normalized_waiting_time(self, state: DirectionState) -> float:
        """计算标准化等待时间"""
        if state.total_passengers_served == 0:
            return 0.0
        avg_waiting = state.total_waiting_time / state.total_passengers_served
        return min(avg_waiting / 30.0, 1.0)  # 标准化到[0,1]，30分钟为上限
        
    def _get_current_state(self) -> Dict[str, Any]:
        """获取当前双向状态"""
        hour = self.current_time // 60
        minute = self.current_time % 60
        
        # 计算双向特征
        up_capacity_util = self._calculate_capacity_utilization(self.up_state)
        down_capacity_util = self._calculate_capacity_utilization(self.down_state)
        
        up_waiting_time = self._calculate_normalized_waiting_time(self.up_state)
        down_waiting_time = self._calculate_normalized_waiting_time(self.down_state)
        
        # 双向约束特征
        departure_count_diff = self.up_state.total_departures - self.down_state.total_departures
        total_buses = len(self.up_state.buses_in_service) + len(self.down_state.buses_in_service)
        
        return {
            # 时间特征
            'time_hour': hour,
            'time_minute': minute,
            
            # 上行特征
            'up_capacity_utilization': up_capacity_util,
            'up_waiting_time': up_waiting_time,
            'up_stranded_passengers': min(self.up_state.stranded_passengers / 100.0, 1.0),
            'up_departure_count': self.up_state.total_departures,
            
            # 下行特征
            'down_capacity_utilization': down_capacity_util,
            'down_waiting_time': down_waiting_time,
            'down_stranded_passengers': min(self.down_state.stranded_passengers / 100.0, 1.0),
            'down_departure_count': self.down_state.total_departures,
            
            # 双向约束特征
            'departure_count_diff': departure_count_diff,
            'total_buses_in_service': total_buses,
            
            # 原始数据（用于调试）
            'current_time': self.current_time,
            'up_waiting_passengers': self.up_state.waiting_passengers,
            'down_waiting_passengers': self.down_state.waiting_passengers
        }
        
    def is_done(self) -> bool:
        """检查是否结束"""
        return self.current_time >= self.service_end
        
    def get_last_departure_intervals(self) -> Dict[str, int]:
        """获取上次发车间隔"""
        return {
            'up': self.current_time - self.up_state.last_departure_time,
            'down': self.current_time - self.down_state.last_departure_time
        }
        
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'up_statistics': {
                'total_departures': self.up_state.total_departures,
                'total_passengers_served': self.up_state.total_passengers_served,
                'total_waiting_time': self.up_state.total_waiting_time,
                'stranded_passengers': self.up_state.stranded_passengers,
                'capacity_utilization': self._calculate_capacity_utilization(self.up_state)
            },
            'down_statistics': {
                'total_departures': self.down_state.total_departures,
                'total_passengers_served': self.down_state.total_passengers_served,
                'total_waiting_time': self.down_state.total_waiting_time,
                'stranded_passengers': self.down_state.stranded_passengers,
                'capacity_utilization': self._calculate_capacity_utilization(self.down_state)
            },
            'bidirectional_constraints': {
                'departure_count_difference': abs(self.up_state.total_departures - self.down_state.total_departures),
                'departure_count_equal': self.up_state.total_departures == self.down_state.total_departures,
                'total_departures': self.up_state.total_departures + self.down_state.total_departures
            }
        }
