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
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
class Passenger:
    """乘客"""
    arrival_time: int  # 到达站点的时间

@dataclass
class DirectionState:
    """单方向状态"""
    buses_in_service: List[BidirectionalBusTrip]
    waiting_passengers_queue: List[Passenger]  # 等待队列（包含每个乘客的到达时间）
    stranded_passengers: int
    total_departures: int
    total_passengers_served: int
    total_waiting_time: float
    last_departure_time: int
    
    @property
    def waiting_passengers(self) -> int:
        """等待乘客数量"""
        return len(self.waiting_passengers_queue)

class BidirectionalBusEnvironment:
    """双向公交系统仿真环境"""
    
    def __init__(self, 
                 service_start: int = 360,      # 服务开始时间 (6:00 AM)
                 service_end: int = 1320,       # 服务结束时间 (10:00 PM)
                 num_stations: int = 20,        # 单方向站点数量
                 bus_capacity: int = 48,        # 公交车容量 (32座位 * 1.5)
                 avg_travel_time: int = 30,     # 平均单程时间(分钟)
                 enable_logging: bool = False): # 是否启用详细日志
        
        self.service_start = service_start
        self.service_end = service_end
        self.num_stations = num_stations
        self.bus_capacity = bus_capacity
        self.avg_travel_time = avg_travel_time
        self.enable_logging = enable_logging
        
        # 当前状态
        self.current_time = service_start
        
        # 监控数据
        self.episode_history = []
        self.action_history = []
        
        # 双向状态
        self.up_state = DirectionState(
            buses_in_service=[],
            waiting_passengers_queue=[],
            stranded_passengers=0,
            total_departures=0,
            total_passengers_served=0,
            total_waiting_time=0.0,
            last_departure_time=-999
        )
        
        self.down_state = DirectionState(
            buses_in_service=[],
            waiting_passengers_queue=[],
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
        """初始化双向乘客流量数据（使用模拟数据）"""
        # 如果已经加载了真实数据，则不使用模拟数据
        if self.passenger_flows:
            return
            
        for minute in range(self.service_start, self.service_end + 1):
            hour = minute // 60
            
            # 根据时间段和方向设置不同的乘客流量强度
            # 调整为更接近真实情况的流量（降低约3倍）
            if 7 <= hour <= 9:  # 早高峰 - 上行流量大
                up_base_flow = 13  # 原40，降低到1/3
                down_base_flow = 5   # 原15，降低到1/3
            elif 17 <= hour <= 19:  # 晚高峰 - 下行流量大
                up_base_flow = 5    # 原15，降低到1/3
                down_base_flow = 13  # 原40，降低到1/3
            elif 10 <= hour <= 16:  # 平峰期
                up_base_flow = 7    # 原20，降低到1/3
                down_base_flow = 7   # 原20，降低到1/3
            else:  # 低峰期
                up_base_flow = 3    # 原8，降低到1/3
                down_base_flow = 3   # 原8，降低到1/3
                
            # 添加随机波动
            up_flow = max(0, int(up_base_flow * random.gauss(1.0, 0.3)))
            down_flow = max(0, int(down_base_flow * random.gauss(1.0, 0.3)))
            
            self.passenger_flows[minute] = {
                'up': up_flow,
                'down': down_flow
            }
    
    def load_passenger_data(self, route_data: Dict[str, Dict[str, pd.DataFrame]]):
        """
        加载真实的乘客数据
        
        Args:
            route_data: 包含'passenger'键的字典，其值为包含'direction_0'和'direction_1'的DataFrame字典
                       每个DataFrame包含: Label, Boarding time, Boarding station, Alighting station, Arrival time
        """
        import pandas as pd
        
        # 清空现有的模拟数据
        self.passenger_flows = {}
        
        # 初始化所有时间点的流量为0
        for minute in range(self.service_start, self.service_end + 1):
            self.passenger_flows[minute] = {'up': 0, 'down': 0}
        
        # 获取乘客数据
        passenger_data = route_data.get('passenger', {})
        
        # 处理上行数据 (direction_0)
        if 'direction_0' in passenger_data:
            df_up = passenger_data['direction_0']
            # 按上车时间统计每分钟的乘客数
            boarding_counts = df_up.groupby('Boarding time').size()
            for time, count in boarding_counts.items():
                time_int = int(time)
                if self.service_start <= time_int <= self.service_end:
                    self.passenger_flows[time_int]['up'] = int(count)
        
        # 处理下行数据 (direction_1)
        if 'direction_1' in passenger_data:
            df_down = passenger_data['direction_1']
            # 按上车时间统计每分钟的乘客数
            boarding_counts = df_down.groupby('Boarding time').size()
            for time, count in boarding_counts.items():
                time_int = int(time)
                if self.service_start <= time_int <= self.service_end:
                    self.passenger_flows[time_int]['down'] = int(count)
        
        if self.enable_logging:
            total_up = sum(flow['up'] for flow in self.passenger_flows.values())
            total_down = sum(flow['down'] for flow in self.passenger_flows.values())
            logger.info(f"加载真实乘客数据: 上行{total_up}人, 下行{total_down}人")
            
    def reset(self) -> Dict[str, Any]:
        """重置环境"""
        self.current_time = self.service_start
        self.trip_counter = 0
        
        # 重置双向状态
        self.up_state = DirectionState(
            buses_in_service=[],
            waiting_passengers_queue=[],
            stranded_passengers=0,
            total_departures=0,
            total_passengers_served=0,
            total_waiting_time=0.0,
            last_departure_time=-999
        )
        
        self.down_state = DirectionState(
            buses_in_service=[],
            waiting_passengers_queue=[],
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
        
        # 记录动作
        self.action_history.append({
            'time': self.current_time,
            'action': action,
            'up_waiting': self.up_state.waiting_passengers,
            'down_waiting': self.down_state.waiting_passengers
        })
        
        # 执行发车决策
        if a_up == 1:
            self._dispatch_bus('up')
            if self.enable_logging:
                logger.info(f"时间{self.current_time}: 上行发车")
            
        if a_down == 1:
            self._dispatch_bus('down')
            if self.enable_logging:
                logger.info(f"时间{self.current_time}: 下行发车")
            
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
        """发车 - 按照论文公式2.6计算等待时间"""
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
        
        # 计算能上车的乘客数量
        passengers_to_board = min(len(state.waiting_passengers_queue), self.bus_capacity)
        new_trip.passenger_count = passengers_to_board
        
        # 按照论文公式2.6计算等待时间
        # Wₘ = Σ Σ (t_b^(m,i,k) - t_a^(m,i,k))
        # 对于每个上车的乘客，计算其实际等待时间
        for i in range(passengers_to_board):
            passenger = state.waiting_passengers_queue[i]
            # 等待时间 = 上车时间 - 到达时间
            waiting_time = self.current_time - passenger.arrival_time
            state.total_waiting_time += waiting_time
        
        # 从队列中移除已上车的乘客
        state.waiting_passengers_queue = state.waiting_passengers_queue[passengers_to_board:]
        
        # 注意：滞留乘客在服务结束时统计（get_statistics中计算）
        
        # 更新统计信息
        state.buses_in_service.append(new_trip)
        state.total_departures += 1
        state.total_passengers_served += passengers_to_board
        state.last_departure_time = self.current_time
            
    def _update_passenger_flow(self):
        """更新双向乘客流量"""
        if self.current_time in self.passenger_flows:
            flows = self.passenger_flows[self.current_time]
            # 添加新到达的乘客到等待队列
            for _ in range(flows['up']):
                self.up_state.waiting_passengers_queue.append(Passenger(arrival_time=self.current_time))
            for _ in range(flows['down']):
                self.down_state.waiting_passengers_queue.append(Passenger(arrival_time=self.current_time))
            
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
        """计算运力利用率 (o_m)"""
        if state.total_departures == 0:
            return 0.0
        return min(state.total_passengers_served / (state.total_departures * self.bus_capacity), 1.0)
    
    def _calculate_load_factor(self, state: DirectionState) -> float:
        """计算满载率 - 当前在途车辆的平均载客率"""
        if len(state.buses_in_service) == 0:
            return 0.0
        total_load = sum(bus.passenger_count / bus.capacity for bus in state.buses_in_service)
        return min(total_load / len(state.buses_in_service), 1.0)
        
    def _calculate_normalized_waiting_time(self, state: DirectionState) -> float:
        """计算标准化等待时间"""
        if state.total_passengers_served == 0:
            return 0.0
        avg_waiting = state.total_waiting_time / state.total_passengers_served
        return min(avg_waiting / 30.0, 1.0)  # 标准化到[0,1]，30分钟为上限
    
    def _normalize_departure_count(self, count: int) -> float:
        """归一化发车次数到[0,1]区间"""
        # 假设一天最多发车100次
        max_departures = 100
        return min(count / max_departures, 1.0)
        
    def _get_current_state(self) -> Dict[str, Any]:
        """获取当前双向状态
        
        根据论文方程(2.1)，返回10维状态空间:
        1. a1_m: 标准化小时
        2. a2_m: 标准化分钟
        3. x1_m: 上行满载率
        4. x2_m: 上行等待时间
        5. x3_m: 上行运力利用率
        6. x4_m: 上行发车次数
        7. y1_m: 下行满载率
        8. y2_m: 下行等待时间
        9. y3_m: 下行运力利用率
        10. y4_m: 下行发车次数
        """
        hour = self.current_time // 60
        minute = self.current_time % 60
        
        # 计算上行特征
        up_load_factor = self._calculate_load_factor(self.up_state)
        up_waiting_time = self._calculate_normalized_waiting_time(self.up_state)
        up_capacity_util = self._calculate_capacity_utilization(self.up_state)
        up_departure_count_norm = self._normalize_departure_count(self.up_state.total_departures)
        
        # 计算下行特征
        down_load_factor = self._calculate_load_factor(self.down_state)
        down_waiting_time = self._calculate_normalized_waiting_time(self.down_state)
        down_capacity_util = self._calculate_capacity_utilization(self.down_state)
        down_departure_count_norm = self._normalize_departure_count(self.down_state.total_departures)
        
        return {
            # 时间特征
            'time_hour': hour,
            'time_minute': minute,
            
            # 上行特征 (论文方程2.1)
            'up_load_factor': up_load_factor,                    # x1_m
            'up_waiting_time': up_waiting_time,                  # x2_m
            'up_capacity_utilization': up_capacity_util,         # x3_m
            'up_departure_count_norm': up_departure_count_norm,  # x4_m
            'up_departure_count': self.up_state.total_departures,
            
            # 下行特征 (论文方程2.1)
            'down_load_factor': down_load_factor,                      # y1_m
            'down_waiting_time': down_waiting_time,                    # y2_m
            'down_capacity_utilization': down_capacity_util,           # y3_m
            'down_departure_count_norm': down_departure_count_norm,    # y4_m
            'down_departure_count': self.down_state.total_departures,
            
            # 额外信息（用于奖励计算和调试）
            'up_stranded_passengers': min(self.up_state.stranded_passengers / 100.0, 1.0),
            'down_stranded_passengers': min(self.down_state.stranded_passengers / 100.0, 1.0),
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
        # 滞留乘客 = 服务结束时仍在等待的乘客
        up_stranded = len(self.up_state.waiting_passengers_queue)
        down_stranded = len(self.down_state.waiting_passengers_queue)
        
        return {
            'up_statistics': {
                'total_departures': self.up_state.total_departures,
                'total_passengers_served': self.up_state.total_passengers_served,
                'total_waiting_time': self.up_state.total_waiting_time,
                'stranded_passengers': up_stranded,
                'capacity_utilization': self._calculate_capacity_utilization(self.up_state)
            },
            'down_statistics': {
                'total_departures': self.down_state.total_departures,
                'total_passengers_served': self.down_state.total_passengers_served,
                'total_waiting_time': self.down_state.total_waiting_time,
                'stranded_passengers': down_stranded,
                'capacity_utilization': self._calculate_capacity_utilization(self.down_state)
            },
            'bidirectional_constraints': {
                'departure_count_difference': abs(self.up_state.total_departures - self.down_state.total_departures),
                'departure_count_equal': self.up_state.total_departures == self.down_state.total_departures,
                'total_departures': self.up_state.total_departures + self.down_state.total_departures
            }
        }
