"""公交仿真环境"""
import numpy as np
from typing import Tuple, Dict, List
from collections import defaultdict
from environment.entities import Bus, Passenger
from data.data_loader import PassengerDataLoader, TrafficDataLoader
from utils.config import RouteConfig


class BusEnvironment:
    """公交仿真环境"""
    
    def __init__(self, config: RouteConfig, passenger_loader: PassengerDataLoader,
                 traffic_loader: TrafficDataLoader):
        """
        初始化环境
        
        Args:
            config: 线路配置
            passenger_loader: 乘客数据加载器
            traffic_loader: 交通数据加载器
        """
        self.config = config
        self.passenger_loader = passenger_loader
        self.traffic_loader = traffic_loader
        
        # 线路信息
        self.route_id = config.route_id
        self.direction = config.direction
        self.num_stations = config.num_stations
        self.start_time = config.start_time
        self.end_time = config.end_time
        self.max_capacity = config.max_capacity
        self.seats = config.seats
        self.alpha = config.alpha
        
        # 奖励函数参数
        self.omega = config.omega
        self.beta = config.beta
        self.zeta = config.zeta
        self.mu = config.mu
        self.delta = config.delta
        
        # 发车约束
        self.t_min = config.t_min
        self.t_max = config.t_max
        
        # 状态变量
        self.current_time = self.start_time
        self.buses = []  # 运行中的公交车列表
        self.dispatch_count_up = 0
        self.dispatch_count_down = 0
        self.last_dispatch_time_up = 0
        self.last_dispatch_time_down = 0
        
        # 乘客管理
        self.waiting_passengers = defaultdict(list)  # {station: [passengers]}
        self.stranded_passengers = []
        self.total_waiting_time = 0
        self.total_passengers_served = 0
        
        # 统计信息
        self.episode_info = {
            'total_reward': 0,
            'dispatch_count_up': 0,
            'dispatch_count_down': 0,
            'stranded_count': 0,
            'avg_waiting_time': 0
        }
        
    def reset(self) -> np.ndarray:
        """
        重置环境
        
        Returns:
            初始状态向量
        """
        self.current_time = self.start_time
        self.buses = []
        self.dispatch_count_up = 0
        self.dispatch_count_down = 0
        self.last_dispatch_time_up = 0
        self.last_dispatch_time_down = 0
        
        self.waiting_passengers = defaultdict(list)
        self.stranded_passengers = []
        self.total_waiting_time = 0
        self.total_passengers_served = 0
        
        self.episode_info = {
            'total_reward': 0,
            'dispatch_count_up': 0,
            'dispatch_count_down': 0,
            'stranded_count': 0,
            'avg_waiting_time': 0
        }
        
        return self._get_state()
    
    def step(self, action: Tuple[int, int]) -> Tuple[np.ndarray, float, bool, dict]:
        """
        执行一步仿真
        
        Args:
            action: (a_up, a_down) 上下行是否发车
            
        Returns:
            state: 下一状态
            reward: 奖励值
            done: 是否结束
            info: 额外信息
        """
        a_up, a_down = action
        
        # 应用发车间隔约束
        if self.current_time - self.last_dispatch_time_up < self.t_min:
            a_up = 0
        if self.current_time - self.last_dispatch_time_up > self.t_max:
            a_up = 1
            
        if self.current_time - self.last_dispatch_time_down < self.t_min:
            a_down = 0
        if self.current_time - self.last_dispatch_time_down > self.t_max:
            a_down = 1
        
        # 更新乘客（添加新到达的乘客）
        self._update_passengers()
        
        # 发车
        if a_up == 1:
            self._dispatch_bus(direction=0)
            self.last_dispatch_time_up = self.current_time
            self.dispatch_count_up += 1
            
        if a_down == 1:
            self._dispatch_bus(direction=1)
            self.last_dispatch_time_down = self.current_time
            self.dispatch_count_down += 1
        
        # 更新所有运行中的公交车
        self._update_buses()
        
        # 计算奖励
        reward = self._calculate_reward((a_up, a_down))
        self.episode_info['total_reward'] += reward
        
        # 前进到下一分钟
        self.current_time += 1
        
        # 判断是否结束
        done = self.current_time > self.end_time
        
        # 获取下一状态
        next_state = self._get_state()
        
        # 额外信息
        info = {
            'current_time': self.current_time,
            'dispatch_count_up': self.dispatch_count_up,
            'dispatch_count_down': self.dispatch_count_down,
            'stranded_count': len(self.stranded_passengers),
            'buses_running': len(self.buses)
        }
        
        return next_state, reward, done, info
    
    def _get_state(self) -> np.ndarray:
        """
        计算当前状态向量
        
        Returns:
            10维状态向量
        """
        # 时间状态
        hour = self.current_time // 60
        minute = self.current_time % 60
        a1 = hour / 24.0
        a2 = minute / 60.0
        
        # 上行状态
        x1 = self._compute_max_load_ratio(direction=0)
        x2 = self._compute_waiting_time(direction=0) / self.mu
        x3 = self._compute_capacity_utilization(direction=0)
        x4 = (self.dispatch_count_up - self.dispatch_count_down) / self.delta
        
        # 下行状态
        y1 = self._compute_max_load_ratio(direction=1)
        y2 = self._compute_waiting_time(direction=1) / self.mu
        y3 = self._compute_capacity_utilization(direction=1)
        y4 = (self.dispatch_count_down - self.dispatch_count_up) / self.delta
        
        state = np.array([a1, a2, x1, x2, x3, x4, y1, y2, y3, y4], dtype=np.float32)
        
        # 确保所有值在合理范围内
        state = np.clip(state, -10, 10)
        
        return state
    
    def _compute_max_load_ratio(self, direction: int) -> float:
        """
        计算满载率（最大断面客流/最大载客量）
        
        Args:
            direction: 方向
            
        Returns:
            满载率
        """
        if not self.buses:
            return 0.0
        
        # 找到该方向的所有公交车
        direction_buses = [bus for bus in self.buses if bus.direction == direction]
        
        if not direction_buses:
            return 0.0
        
        # 计算最大载客量
        max_load = max([bus.get_load() for bus in direction_buses])
        
        return min(max_load / self.max_capacity, 1.0)
    
    def _compute_waiting_time(self, direction: int) -> float:
        """
        计算总等待时间
        
        Args:
            direction: 方向
            
        Returns:
            总等待时间
        """
        total_waiting = 0.0
        
        # 计算所有等待乘客的等待时间
        for station, passengers in self.waiting_passengers.items():
            for passenger in passengers:
                if passenger.boarding_station >= 0:  # 确保是有效乘客
                    waiting = self.current_time - passenger.arrival_time
                    total_waiting += max(0, waiting)
        
        return total_waiting
    
    def _compute_capacity_utilization(self, direction: int) -> float:
        """
        计算客运容量利用率
        
        Args:
            direction: 方向
            
        Returns:
            容量利用率
        """
        # 实际消耗容量
        o_m = 0
        # 提供的容量
        e_m = self.alpha * self.seats * (self.num_stations - 1)
        
        if e_m == 0:
            return 0.0
        
        # 计算实际消耗（简化版本：使用当前车辆载客量）
        direction_buses = [bus for bus in self.buses if bus.direction == direction]
        for bus in direction_buses:
            o_m += bus.get_load()
        
        return min(o_m / e_m, 1.0) if e_m > 0 else 0.0
    
    def _calculate_reward(self, action: Tuple[int, int]) -> float:
        """
        计算奖励值
        
        Args:
            action: (a_up, a_down)
            
        Returns:
            总奖励
        """
        a_up, a_down = action
        
        # 上行奖励
        o_up = self._compute_actual_capacity(direction=0)
        e_up = self._compute_provided_capacity(direction=0)
        W_up = self._compute_waiting_time(direction=0)
        d_up = self._count_stranded_passengers(direction=0)
        
        if e_up > 0:
            capacity_ratio_up = o_up / e_up
        else:
            capacity_ratio_up = 0.0
        
        if a_up == 0:  # 不发车
            r_up = 1 - capacity_ratio_up - (self.omega * W_up) - (self.beta * d_up) + \
                   self.zeta * (self.dispatch_count_up - self.dispatch_count_down)
        else:  # 发车
            r_up = capacity_ratio_up - (self.beta * d_up) - \
                   self.zeta * (self.dispatch_count_up - self.dispatch_count_down)
        
        # 下行奖励
        o_down = self._compute_actual_capacity(direction=1)
        e_down = self._compute_provided_capacity(direction=1)
        W_down = self._compute_waiting_time(direction=1)
        d_down = self._count_stranded_passengers(direction=1)
        
        if e_down > 0:
            capacity_ratio_down = o_down / e_down
        else:
            capacity_ratio_down = 0.0
        
        if a_down == 0:  # 不发车
            r_down = 1 - capacity_ratio_down - (self.omega * W_down) - (self.beta * d_down) - \
                     self.zeta * (self.dispatch_count_up - self.dispatch_count_down)
        else:  # 发车
            r_down = capacity_ratio_down - (self.beta * d_down) + \
                     self.zeta * (self.dispatch_count_up - self.dispatch_count_down)
        
        return r_up + r_down
    
    def _compute_actual_capacity(self, direction: int) -> float:
        """计算实际消耗容量"""
        total = 0.0
        direction_buses = [bus for bus in self.buses if bus.direction == direction]
        for bus in direction_buses:
            total += bus.get_load()
        return total
    
    def _compute_provided_capacity(self, direction: int) -> float:
        """计算提供的容量"""
        return self.alpha * self.seats * (self.num_stations - 1)
    
    def _count_stranded_passengers(self, direction: int) -> int:
        """统计滞留乘客数量"""
        return len([p for p in self.stranded_passengers if p.is_stranded])
    
    def _dispatch_bus(self, direction: int):
        """
        发车
        
        Args:
            direction: 方向
        """
        bus_id = len(self.buses)
        bus = Bus(
            bus_id=bus_id,
            direction=direction,
            current_station=0,
            capacity=self.max_capacity,
            dispatch_time=self.current_time,
            current_time=self.current_time
        )
        self.buses.append(bus)
    
    def _update_buses(self):
        """更新所有运行中的公交车状态"""
        buses_to_remove = []
        
        for bus in self.buses:
            # 检查是否到达终点
            if bus.current_station >= self.num_stations - 1:
                buses_to_remove.append(bus)
                continue
            
            # 乘客下车
            alighting = bus.alight_passengers(bus.current_station)
            self.total_passengers_served += len(alighting)
            
            # 乘客上车
            waiting = self.waiting_passengers[bus.current_station]
            if waiting:
                stranded = bus.board_passengers(waiting, self.current_time)
                self.stranded_passengers.extend(stranded)
                self.waiting_passengers[bus.current_station] = stranded
            
            # 移动到下一站
            travel_time = self.traffic_loader.get_next_station_time(
                bus.current_station, self.current_time
            )
            bus.move_to_next_station(travel_time)
        
        # 移除已到达终点的公交车
        for bus in buses_to_remove:
            self.buses.remove(bus)
    
    def _update_passengers(self):
        """更新乘客状态，添加新到达的乘客"""
        new_passengers = self.passenger_loader.get_passengers_at_time(self.current_time)
        
        for passenger in new_passengers:
            station = passenger.boarding_station
            self.waiting_passengers[station].append(passenger)
