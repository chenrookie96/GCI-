# src/environment/station_level_simulator.py
"""
站点级别公交系统仿真环境
完整实现论文中的多站点模型，乘客在不同站点上下车
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from collections import deque, defaultdict
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Passenger:
    """乘客 - 增强版，包含站点信息"""
    passenger_id: int                    # 乘客ID
    arrival_time: int                    # 到达站点的时间(分钟)
    boarding_station: int                # 上车站点ID
    alighting_station: int               # 下车站点ID
    direction: str                       # 方向 ('up' or 'down')
    boarding_time: Optional[int] = None  # 实际上车时间


@dataclass
class Station:
    """公交站点 - 优化版本，使用deque提升性能"""
    station_id: int                      # 站点ID
    station_name: str                    # 站点名称
    waiting_passengers: Any = field(default_factory=deque)  # 等待队列（使用deque）
    
    def __post_init__(self):
        """确保waiting_passengers是deque类型"""
        if not isinstance(self.waiting_passengers, deque):
            self.waiting_passengers = deque(self.waiting_passengers)
    
    def add_passenger(self, passenger: Passenger):
        """添加乘客到等待队列 - O(1)"""
        self.waiting_passengers.append(passenger)
    
    def remove_passengers(self, count: int) -> List[Passenger]:
        """移除指定数量的乘客（FIFO）- O(k) where k=count"""
        boarding = []
        for _ in range(min(count, len(self.waiting_passengers))):
            boarding.append(self.waiting_passengers.popleft())
        return boarding
    
    @property
    def waiting_count(self) -> int:
        """等待乘客数量 - O(1)"""
        return len(self.waiting_passengers)


@dataclass
class BusTrip:
    """公交行程 - 增强版，包含站点列表"""
    trip_id: int                         # 行程ID
    departure_time: int                  # 起点站发车时间
    direction: str                       # 方向 ('up' or 'down')
    capacity: int                        # 车辆容量
    stations: List[int]                  # 经过的站点ID列表
    current_passengers: List[Passenger] = field(default_factory=list)  # 当前车上的乘客
    station_arrival_times: Dict[int, int] = field(default_factory=dict)  # 各站点到达时间
    current_station_index: int = 0       # 当前站点索引
    completed: bool = False              # 是否完成行程
    
    @property
    def current_load(self) -> int:
        """当前载客数"""
        return len(self.current_passengers)
    
    @property
    def available_capacity(self) -> int:
        """可用容量"""
        return self.capacity - self.current_load
    
    @property
    def current_station(self) -> Optional[int]:
        """当前站点ID"""
        if self.current_station_index < len(self.stations):
            return self.stations[self.current_station_index]
        return None
    
    @property
    def next_station(self) -> Optional[int]:
        """下一个站点ID"""
        next_index = self.current_station_index + 1
        if next_index < len(self.stations):
            return self.stations[next_index]
        return None


@dataclass
class DirectionState:
    """单方向状态 - 增强版，包含站点列表"""
    direction: str                       # 方向标识
    stations: List[Station]              # 该方向的所有站点
    buses_in_service: List[BusTrip] = field(default_factory=list)
    total_departures: int = 0
    fixed_departures: int = 0            # 固定发车次数（首班车和末班车）
    drl_departures: int = 0              # DRL控制发车次数
    total_passengers_served: int = 0
    total_waiting_time: float = 0.0      # 所有乘客的累计等待时间(分钟)
    last_departure_time: int = -999
    first_bus_dispatched: bool = False   # 首班车是否已发
    last_bus_dispatched: bool = False    # 末班车是否已发
    
    @property
    def stranded_passengers(self) -> int:
        """滞留乘客：服务结束时所有站点的等待乘客总数"""
        return sum(station.waiting_count for station in self.stations)
    
    @property
    def total_waiting_passengers(self) -> int:
        """当前所有站点的等待乘客总数"""
        return self.stranded_passengers
    
    def get_station(self, station_id: int) -> Optional[Station]:
        """根据ID获取站点"""
        for station in self.stations:
            if station.station_id == station_id:
                return station
        return None


class StationLevelBusEnvironment:
    """站点级别公交系统仿真环境"""
    
    def __init__(self,
                 service_start: int = 360,      # 服务开始时间 (6:00 AM)
                 service_end: int = 1260,       # 服务结束时间 (9:00 PM)
                 num_stations: int = 20,        # 单方向站点数量
                 bus_capacity: int = 48,        # 公交车容量
                 dwell_time: float = 0.5,       # 停靠时间(分钟)
                 avg_travel_time: float = 1.5,  # 平均站间行驶时间(分钟)
                 enable_logging: bool = False):
        
        self.service_start = service_start
        self.service_end = service_end
        self.num_stations = num_stations
        self.bus_capacity = bus_capacity
        self.dwell_time = dwell_time
        self.avg_travel_time = avg_travel_time
        self.enable_logging = enable_logging
        
        # 当前状态
        self.current_time = service_start
        self.trip_counter = 0
        
        # 双向状态
        self.up_state = None
        self.down_state = None
        
        # 固定发车状态跟踪
        self.action_overridden = False
        self.fixed_dispatch_up = False
        self.fixed_dispatch_down = False
        
        # 乘客数据：按时间和站点组织
        self.passenger_arrivals = {}  # {time: {station_id: [passengers]}}
        
        # 站点间行驶时间
        self.travel_times = {}  # {(from_station, to_station): time}
        
        # 初始化
        self._initialize_states()
        
    def _initialize_states(self):
        """初始化双向状态"""
        # 上行：站点1 -> 站点num_stations
        up_stations = [
            Station(station_id=i, station_name=f"Up_Station_{i}")
            for i in range(1, self.num_stations + 1)
        ]
        self.up_state = DirectionState(direction='up', stations=up_stations)
        
        # 下行：站点num_stations -> 站点1
        down_stations = [
            Station(station_id=i, station_name=f"Down_Station_{i}")
            for i in range(self.num_stations, 0, -1)
        ]
        self.down_state = DirectionState(direction='down', stations=down_stations)
        
    def load_passenger_data(self, route_data: Dict[str, Dict[str, pd.DataFrame]],
                           filter_out_of_service: bool = True,
                           convert_station_ids: bool = True):
        """
        加载真实的乘客数据并分配到各站点
        
        Args:
            route_data: 包含'passenger'键的字典，其值为包含'direction_0'和'direction_1'的DataFrame字典
                       每个DataFrame包含: Label, Boarding time, Boarding station, Alighting station, Arrival time
            filter_out_of_service: 是否过滤服务时间外的乘客（默认True）
            convert_station_ids: 是否转换站点ID从0-based到1-based（默认True）
        """
        passenger_data = route_data.get('passenger', {})
        
        # 清空现有数据
        self.passenger_arrivals = {}
        
        if self.enable_logging:
            logger.info(f"加载乘客数据 (filter_out_of_service={filter_out_of_service}, convert_station_ids={convert_station_ids})")
        
        # 处理上行数据 (direction_0)
        if 'direction_0' in passenger_data:
            df_up = passenger_data['direction_0']
            if self.enable_logging:
                logger.info(f"处理上行乘客数据: {len(df_up)} 条原始记录")
            self._process_passenger_dataframe(df_up, 'up', filter_out_of_service, convert_station_ids)
        
        # 处理下行数据 (direction_1)
        if 'direction_1' in passenger_data:
            df_down = passenger_data['direction_1']
            if self.enable_logging:
                logger.info(f"处理下行乘客数据: {len(df_down)} 条原始记录")
            self._process_passenger_dataframe(df_down, 'down', filter_out_of_service, convert_station_ids)
        
        # 统计加载结果
        total_up = sum(len(arrivals.get('up', [])) for arrivals in self.passenger_arrivals.values())
        total_down = sum(len(arrivals.get('down', [])) for arrivals in self.passenger_arrivals.values())
        if self.enable_logging:
            logger.info(f"乘客数据加载完成: 上行{total_up}人, 下行{total_down}人, 总计{total_up+total_down}人")
        
        # 加载交通数据（站点间行驶时间）
        traffic_data = route_data.get('traffic', {})
        if traffic_data:
            self._load_travel_times(traffic_data)
    
    def _process_passenger_dataframe(self, df: pd.DataFrame, direction: str, 
                                     filter_out_of_service: bool = True,
                                     convert_station_ids: bool = True):
        """
        处理乘客DataFrame，转换为Passenger对象并按时间和站点组织
        优化：使用向量化操作和批量处理
        
        Args:
            df: 乘客数据DataFrame
            direction: 方向 ('up' or 'down')
            filter_out_of_service: 是否过滤服务时间外的乘客
            convert_station_ids: 是否转换站点ID（从0-based转为1-based）
        """
        filtered_time_count = 0
        filtered_station_count = 0
        passengers = []
        
        # 记录原始站点ID范围
        orig_boarding_min = df['Boarding station'].min()
        orig_boarding_max = df['Boarding station'].max()
        orig_alighting_min = df['Alighting station'].min()
        orig_alighting_max = df['Alighting station'].max()
        
        for _, row in df.iterrows():
            arrival_time = int(row['Boarding time'])
            boarding_station = int(row['Boarding station'])
            alighting_station = int(row['Alighting station'])
            
            # 时间过滤
            if filter_out_of_service:
                if arrival_time < self.service_start or arrival_time >= self.service_end:
                    filtered_time_count += 1
                    continue
            
            # 站点ID转换（从0-based转为1-based）
            if convert_station_ids:
                boarding_station += 1
                alighting_station += 1
            
            # 验证站点ID有效性
            num_stations = len(self.up_state.stations) if direction == 'up' else len(self.down_state.stations)
            if boarding_station < 1 or boarding_station > num_stations:
                if self.enable_logging:
                    logger.warning(f"Invalid boarding station {boarding_station} for {direction} direction (valid: 1-{num_stations})")
                filtered_station_count += 1
                continue
            if alighting_station < 1 or alighting_station > num_stations:
                if self.enable_logging:
                    logger.warning(f"Invalid alighting station {alighting_station} for {direction} direction (valid: 1-{num_stations})")
                filtered_station_count += 1
                continue
            
            # 创建乘客对象
            passenger = Passenger(
                passenger_id=int(row['Label']),
                arrival_time=arrival_time,
                boarding_station=boarding_station,
                alighting_station=alighting_station,
                direction=direction
            )
            passengers.append(passenger)
        
        # 记录转换后的站点ID范围
        if passengers:
            conv_boarding_min = min(p.boarding_station for p in passengers)
            conv_boarding_max = max(p.boarding_station for p in passengers)
            conv_alighting_min = min(p.alighting_station for p in passengers)
            conv_alighting_max = max(p.alighting_station for p in passengers)
            
            if self.enable_logging:
                logger.info(f"  {direction.capitalize()} 站点ID转换: Boarding [{orig_boarding_min}-{orig_boarding_max}] → [{conv_boarding_min}-{conv_boarding_max}]")
                logger.info(f"  {direction.capitalize()} 站点ID转换: Alighting [{orig_alighting_min}-{orig_alighting_max}] → [{conv_alighting_min}-{conv_alighting_max}]")
        
        # 记录过滤统计
        if self.enable_logging:
            logger.info(f"  {direction.capitalize()} 乘客: 总数{len(df)}, 加载{len(passengers)}, 时间过滤{filtered_time_count}, 站点过滤{filtered_station_count}")
        
        # 批量组织到时间字典
        temp_dict = defaultdict(list)
        for passenger in passengers:
            temp_dict[passenger.arrival_time].append(passenger)
        
        # 合并到主字典
        for arrival_time, passenger_list in temp_dict.items():
            if arrival_time not in self.passenger_arrivals:
                self.passenger_arrivals[arrival_time] = {'up': [], 'down': []}
            self.passenger_arrivals[arrival_time][direction].extend(passenger_list)
    
    def _load_travel_times(self, traffic_data: Dict[str, pd.DataFrame]):
        """
        加载站点间行驶时间数据
        
        Args:
            traffic_data: 交通数据字典
        """
        # TODO: 从交通数据中提取站点间行驶时间
        # 目前使用平均值
        pass
    
    def validate_passenger_loading(self, expected_counts: Dict[str, int]) -> Dict[str, Any]:
        """
        验证乘客数据是否正确加载
        
        Args:
            expected_counts: 期望的乘客数量 {'up': int, 'down': int}
            
        Returns:
            验证结果字典，包含:
            - up_loaded: 实际加载的上行乘客数
            - down_loaded: 实际加载的下行乘客数
            - up_expected: 期望的上行乘客数
            - down_expected: 期望的下行乘客数
            - match: 是否匹配
            - issues: 问题列表
            - station_id_range: 站点ID范围
            - time_range: 时间范围
        """
        # 统计实际加载的乘客数
        up_loaded = sum(len(arrivals.get('up', [])) for arrivals in self.passenger_arrivals.values())
        down_loaded = sum(len(arrivals.get('down', [])) for arrivals in self.passenger_arrivals.values())
        
        # 收集所有乘客用于验证
        all_passengers = []
        for arrivals in self.passenger_arrivals.values():
            all_passengers.extend(arrivals.get('up', []))
            all_passengers.extend(arrivals.get('down', []))
        
        issues = []
        
        # 检查数量匹配
        up_expected = expected_counts.get('up', 0)
        down_expected = expected_counts.get('down', 0)
        
        if up_loaded != up_expected:
            issues.append(f"上行乘客数不匹配: 期望{up_expected}, 实际{up_loaded}, 差异{up_expected - up_loaded}")
        
        if down_loaded != down_expected:
            issues.append(f"下行乘客数不匹配: 期望{down_expected}, 实际{down_loaded}, 差异{down_expected - down_loaded}")
        
        # 检查站点ID范围
        if all_passengers:
            boarding_stations = [p.boarding_station for p in all_passengers]
            alighting_stations = [p.alighting_station for p in all_passengers]
            
            boarding_min, boarding_max = min(boarding_stations), max(boarding_stations)
            alighting_min, alighting_max = min(alighting_stations), max(alighting_stations)
            
            num_stations = len(self.up_state.stations)
            
            if boarding_min < 1:
                issues.append(f"发现无效的上车站点ID: {boarding_min} < 1")
            if boarding_max > num_stations:
                issues.append(f"发现无效的上车站点ID: {boarding_max} > {num_stations}")
            if alighting_min < 1:
                issues.append(f"发现无效的下车站点ID: {alighting_min} < 1")
            if alighting_max > num_stations:
                issues.append(f"发现无效的下车站点ID: {alighting_max} > {num_stations}")
            
            station_id_range = {
                'boarding': (boarding_min, boarding_max),
                'alighting': (alighting_min, alighting_max)
            }
        else:
            station_id_range = None
            issues.append("没有加载任何乘客")
        
        # 检查时间范围
        if self.passenger_arrivals:
            time_min = min(self.passenger_arrivals.keys())
            time_max = max(self.passenger_arrivals.keys())
            
            if time_min < self.service_start:
                issues.append(f"发现服务开始前的乘客: {time_min} < {self.service_start}")
            if time_max >= self.service_end:
                issues.append(f"发现服务结束后的乘客: {time_max} >= {self.service_end}")
            
            time_range = (time_min, time_max)
        else:
            time_range = None
        
        return {
            'up_loaded': up_loaded,
            'down_loaded': down_loaded,
            'up_expected': up_expected,
            'down_expected': down_expected,
            'match': len(issues) == 0,
            'issues': issues,
            'station_id_range': station_id_range,
            'time_range': time_range
        }
    
    def _get_travel_time(self, from_station: int, to_station: int) -> float:
        """
        获取两个站点间的行驶时间
        
        Args:
            from_station: 起始站点ID
            to_station: 目标站点ID
            
        Returns:
            行驶时间(分钟)
        """
        # 如果有真实数据，使用真实数据
        if (from_station, to_station) in self.travel_times:
            return self.travel_times[(from_station, to_station)]
        
        # 否则使用平均值
        return self.avg_travel_time
    
    def reset(self) -> Dict[str, Any]:
        """重置环境"""
        self.current_time = self.service_start
        self.trip_counter = 0
        
        # 重置固定发车状态跟踪
        self.action_overridden = False
        self.fixed_dispatch_up = False
        self.fixed_dispatch_down = False
        
        # 重新初始化状态
        self._initialize_states()
        
        # 清空所有站点的等待队列
        for station in self.up_state.stations + self.down_state.stations:
            station.waiting_passengers.clear()
        
        return self._get_current_state()
    
    def _calculate_load_factor(self, state: DirectionState) -> float:
        """计算满载率 - 当前在途车辆的平均载客率"""
        if len(state.buses_in_service) == 0:
            return 0.0
        total_load = sum(bus.current_load / bus.capacity for bus in state.buses_in_service)
        return min(total_load / len(state.buses_in_service), 1.0)
    
    def _calculate_capacity_utilization(self, state: DirectionState) -> float:
        """计算运力利用率"""
        if state.total_departures == 0:
            return 0.0
        return min(state.total_passengers_served / (state.total_departures * self.bus_capacity), 1.0)
    
    def _calculate_normalized_waiting_time(self, state: DirectionState) -> float:
        """计算标准化等待时间"""
        if state.total_passengers_served == 0:
            return 0.0
        avg_waiting = state.total_waiting_time / state.total_passengers_served
        return min(avg_waiting / 30.0, 1.0)  # 标准化到[0,1]，30分钟为上限
    
    def _get_current_state(self) -> Dict[str, Any]:
        """获取当前状态（10维状态空间，保持兼容）"""
        hour = self.current_time // 60
        minute = self.current_time % 60
        
        # 计算上行特征
        up_load_factor = self._calculate_load_factor(self.up_state)
        up_waiting_time = self._calculate_normalized_waiting_time(self.up_state)
        up_capacity_util = self._calculate_capacity_utilization(self.up_state)
        up_departure_count_norm = min(self.up_state.total_departures / 100.0, 1.0)
        
        # 计算下行特征
        down_load_factor = self._calculate_load_factor(self.down_state)
        down_waiting_time = self._calculate_normalized_waiting_time(self.down_state)
        down_capacity_util = self._calculate_capacity_utilization(self.down_state)
        down_departure_count_norm = min(self.down_state.total_departures / 100.0, 1.0)
        
        return {
            # 时间特征
            'time_hour': hour,
            'time_minute': minute,
            
            # 上行特征
            'up_load_factor': up_load_factor,
            'up_waiting_time': up_waiting_time,
            'up_capacity_utilization': up_capacity_util,
            'up_departure_count_norm': up_departure_count_norm,
            'up_departure_count': self.up_state.total_departures,
            
            # 下行特征
            'down_load_factor': down_load_factor,
            'down_waiting_time': down_waiting_time,
            'down_capacity_utilization': down_capacity_util,
            'down_departure_count_norm': down_departure_count_norm,
            'down_departure_count': self.down_state.total_departures,
            
            # 额外信息
            'up_stranded_passengers': min(self.up_state.stranded_passengers / 100.0, 1.0),
            'down_stranded_passengers': min(self.down_state.stranded_passengers / 100.0, 1.0),
            'current_time': self.current_time,
            'up_waiting_passengers': self.up_state.total_waiting_passengers,
            'down_waiting_passengers': self.down_state.total_waiting_passengers,
            
            # 固定发车状态
            'action_overridden': self.action_overridden,
            'fixed_dispatch_up': self.fixed_dispatch_up,
            'fixed_dispatch_down': self.fixed_dispatch_down,
            'up_first_bus_dispatched': self.up_state.first_bus_dispatched,
            'down_first_bus_dispatched': self.down_state.first_bus_dispatched
        }
    
    def _update_passenger_arrivals(self):
        """更新乘客到达各站点"""
        if self.current_time not in self.passenger_arrivals:
            return
        
        arrivals = self.passenger_arrivals[self.current_time]
        
        # 上行乘客到达
        for passenger in arrivals.get('up', []):
            station = self.up_state.get_station(passenger.boarding_station)
            if station:
                station.add_passenger(passenger)
        
        # 下行乘客到达
        for passenger in arrivals.get('down', []):
            station = self.down_state.get_station(passenger.boarding_station)
            if station:
                station.add_passenger(passenger)
    
    def _check_fixed_departures(self) -> Tuple[bool, bool]:
        """
        检查是否需要固定发车（首班车和末班车）
        
        Returns:
            (up_fixed, down_fixed): 元组，指示上行和下行是否需要固定发车
        """
        up_fixed = False
        down_fixed = False
        
        # 检查首班车（服务开始时刻）
        if self.current_time == self.service_start:
            if not self.up_state.first_bus_dispatched:
                up_fixed = True
            if not self.down_state.first_bus_dispatched:
                down_fixed = True
        
        # 检查末班车（在服务结束时刻发车）
        if self.current_time == self.service_end:
            if not self.up_state.last_bus_dispatched:
                up_fixed = True
            if not self.down_state.last_bus_dispatched:
                down_fixed = True
        
        return up_fixed, down_fixed
    
    def step(self, action: Tuple[int, int]) -> Tuple[Dict[str, Any], float, bool]:
        """执行一步仿真（保持接口兼容）"""
        a_up, a_down = action
        
        # 1. 检查固定发车（首班车和末班车）
        up_fixed, down_fixed = self._check_fixed_departures()
        
        # 记录固定发车状态
        self.action_overridden = up_fixed or down_fixed
        self.fixed_dispatch_up = up_fixed
        self.fixed_dispatch_down = down_fixed
        
        # 2. 执行发车决策（固定发车优先）
        if up_fixed:
            self._dispatch_bus('up', is_fixed=True)
        elif a_up == 1:
            self._dispatch_bus('up', is_fixed=False)
        
        if down_fixed:
            self._dispatch_bus('down', is_fixed=True)
        elif a_down == 1:
            self._dispatch_bus('down', is_fixed=False)
        
        # 3. 更新时间
        self.current_time += 1
        
        # 4. 更新乘客到达各站点
        self._update_passenger_arrivals()
        
        # 5. 更新在途公交车
        self._update_buses()
        
        # 检查是否结束
        done = self.is_done()
        
        return self._get_current_state(), 0.0, done
    
    def _dispatch_bus(self, direction: str, is_fixed: bool = False):
        """
        发车 - 创建包含所有站点的行程
        
        Args:
            direction: 方向 ('up' or 'down')
            is_fixed: 是否为固定发车（首班车或末班车），默认False表示DRL控制发车
        """
        self.trip_counter += 1
        
        # 获取对应方向的状态
        state = self.up_state if direction == 'up' else self.down_state
        
        # 获取该方向的所有站点ID
        station_ids = [s.station_id for s in state.stations]
        
        # 创建新行程
        trip = BusTrip(
            trip_id=self.trip_counter,
            departure_time=self.current_time,
            direction=direction,
            capacity=self.bus_capacity,
            stations=station_ids,
            current_passengers=[],
            station_arrival_times={},
            current_station_index=0,
            completed=False
        )
        
        # 计算各站点的到达时间
        current_time = self.current_time
        for i, station_id in enumerate(station_ids):
            if i == 0:
                # 起点站：当前时间
                trip.station_arrival_times[station_id] = current_time
            else:
                # 后续站点：累加行驶时间和停靠时间
                travel_time = self._get_travel_time(station_ids[i-1], station_id)
                current_time += travel_time + self.dwell_time
                trip.station_arrival_times[station_id] = int(current_time)
        
        # 在起点站接乘客
        start_station_id = station_ids[0]
        self._process_station_stop(trip, start_station_id, state)
        
        # 添加到在途列表
        state.buses_in_service.append(trip)
        state.total_departures += 1
        state.last_departure_time = self.current_time
        
        # 根据is_fixed参数更新计数器
        if is_fixed:
            state.fixed_departures += 1
            # 设置首班车或末班车标志
            if self.current_time == self.service_start:
                state.first_bus_dispatched = True
            elif self.current_time == self.service_end:
                state.last_bus_dispatched = True
        else:
            state.drl_departures += 1
        
        # 日志中标记发车类型
        if self.enable_logging:
            dispatch_type = "FIXED" if is_fixed else "DRL"
            logger.info(f"时间{self.current_time}: {direction}方向发车({dispatch_type})，行程ID={trip.trip_id}")
    
    def _process_station_stop(self, trip: BusTrip, station_id: int, state: DirectionState):
        """
        处理公交车在站点的停靠
        
        Args:
            trip: 公交行程
            station_id: 站点ID
            state: 方向状态
        """
        station = state.get_station(station_id)
        if not station:
            return
        
        # 1. 乘客下车
        alighting_passengers = [p for p in trip.current_passengers 
                               if p.alighting_station == station_id]
        for passenger in alighting_passengers:
            trip.current_passengers.remove(passenger)
        
        # 2. 乘客上车
        boarding_passengers = []
        available_capacity = trip.available_capacity
        if available_capacity > 0 and station.waiting_count > 0:
            # 从等待队列中取出乘客
            boarding_passengers = station.remove_passengers(available_capacity)
            
            for passenger in boarding_passengers:
                # 记录上车时间
                passenger.boarding_time = self.current_time
                
                # 计算等待时间（论文公式2.6）
                waiting_time = passenger.boarding_time - passenger.arrival_time
                state.total_waiting_time += waiting_time
                state.total_passengers_served += 1
                
                # 乘客上车
                trip.current_passengers.append(passenger)
        
        if self.enable_logging and (alighting_passengers or boarding_passengers):
            logger.info(f"  站点{station_id}: 下车{len(alighting_passengers)}人, 上车{len(boarding_passengers)}人")
    
    def _update_buses(self):
        """更新所有在途公交车"""
        # 更新上行车辆
        for trip in self.up_state.buses_in_service[:]:
            if trip.completed:
                continue
            
            # 检查是否到达下一个站点
            if trip.next_station:
                next_arrival_time = trip.station_arrival_times[trip.next_station]
                if self.current_time >= next_arrival_time:
                    # 到达下一个站点
                    trip.current_station_index += 1
                    current_station = trip.current_station
                    
                    # 处理站点停靠
                    self._process_station_stop(trip, current_station, self.up_state)
                    
                    # 检查是否完成行程
                    if trip.next_station is None:
                        trip.completed = True
        
        # 更新下行车辆
        for trip in self.down_state.buses_in_service[:]:
            if trip.completed:
                continue
            
            # 检查是否到达下一个站点
            if trip.next_station:
                next_arrival_time = trip.station_arrival_times[trip.next_station]
                if self.current_time >= next_arrival_time:
                    # 到达下一个站点
                    trip.current_station_index += 1
                    current_station = trip.current_station
                    
                    # 处理站点停靠
                    self._process_station_stop(trip, current_station, self.down_state)
                    
                    # 检查是否完成行程
                    if trip.next_station is None:
                        trip.completed = True
        
        # 移除已完成的行程
        self.up_state.buses_in_service = [t for t in self.up_state.buses_in_service if not t.completed]
        self.down_state.buses_in_service = [t for t in self.down_state.buses_in_service if not t.completed]
    
    def is_done(self) -> bool:
        """
        检查是否结束
        
        Episode在末班车发出后继续运行，直到末班车有足够时间完成行程
        计算：service_end + 单程时间
        """
        # 计算单程时间
        num_stations = len(self.up_state.stations)
        single_trip_time = int((num_stations - 1) * (self.avg_travel_time + self.dwell_time))
        
        # Episode结束时间 = 服务结束时间 + 单程时间（让末班车完成）
        episode_end_time = self.service_end + single_trip_time
        
        return self.current_time >= episode_end_time
    
    def get_last_departure_intervals(self) -> Dict[str, int]:
        """获取上次发车间隔"""
        return {
            'up': self.current_time - self.up_state.last_departure_time,
            'down': self.current_time - self.down_state.last_departure_time
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息（保持格式兼容）"""
        # 计算上行DRL控制比例
        up_drl_ratio = 0.0
        if self.up_state.total_departures > 0:
            up_drl_ratio = self.up_state.drl_departures / self.up_state.total_departures
        
        # 计算下行DRL控制比例
        down_drl_ratio = 0.0
        if self.down_state.total_departures > 0:
            down_drl_ratio = self.down_state.drl_departures / self.down_state.total_departures
        
        # 获取首班车和末班车时间
        up_first_bus_time = self.service_start if self.up_state.first_bus_dispatched else None
        up_last_bus_time = self.service_end if self.up_state.last_bus_dispatched else None
        down_first_bus_time = self.service_start if self.down_state.first_bus_dispatched else None
        down_last_bus_time = self.service_end if self.down_state.last_bus_dispatched else None
        
        return {
            'up_statistics': {
                'total_departures': self.up_state.total_departures,
                'fixed_departures': self.up_state.fixed_departures,
                'drl_departures': self.up_state.drl_departures,
                'drl_control_ratio': up_drl_ratio,
                'first_bus_time': up_first_bus_time,
                'last_bus_time': up_last_bus_time,
                'total_passengers_served': self.up_state.total_passengers_served,
                'total_waiting_time': self.up_state.total_waiting_time,
                'stranded_passengers': self.up_state.stranded_passengers,
                'capacity_utilization': self._calculate_capacity_utilization(self.up_state)
            },
            'down_statistics': {
                'total_departures': self.down_state.total_departures,
                'fixed_departures': self.down_state.fixed_departures,
                'drl_departures': self.down_state.drl_departures,
                'drl_control_ratio': down_drl_ratio,
                'first_bus_time': down_first_bus_time,
                'last_bus_time': down_last_bus_time,
                'total_passengers_served': self.down_state.total_passengers_served,
                'total_waiting_time': self.down_state.total_waiting_time,
                'stranded_passengers': self.down_state.stranded_passengers,
                'capacity_utilization': self._calculate_capacity_utilization(self.down_state)
            },
            'bidirectional_constraints': {
                'departure_count_difference': abs(self.up_state.total_departures - 
                                                 self.down_state.total_departures),
                'departure_count_equal': (self.up_state.total_departures == 
                                         self.down_state.total_departures),
                'total_departures': (self.up_state.total_departures + 
                                    self.down_state.total_departures)
            }
        }
