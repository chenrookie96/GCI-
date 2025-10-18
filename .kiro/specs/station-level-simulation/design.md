# Design Document

## Overview

本设计文档详细说明如何实现完整的站点级别公交模拟系统。当前实现将所有乘客视为在起点站上车，导致等待时间计算不准确。新设计将模拟公交车沿途停靠各个站点，在每个站点处理乘客上下车，使用真实数据中的站点信息，完全符合论文的模型。

## Architecture

### 当前架构的问题

**当前实现：**
```
发车 → 所有乘客在起点站上车 → 公交车直接到达终点
```

**问题：**
1. 忽略了中间站点
2. 所有乘客的等待时间从起点站发车开始计算
3. 无法准确反映乘客在不同站点的等待情况

**新架构：**
```
发车 → 站点1(上下车) → 站点2(上下车) → ... → 站点K(上下车) → 到达终点
```

### 核心组件

#### 1. Station (站点)
表示公交线路上的一个站点

#### 2. StationQueue (站点队列)
每个站点维护的乘客等待队列

#### 3. BusTrip (公交行程)
包含沿途所有站点的完整行程

#### 4. Passenger (乘客)
包含上车站点、下车站点、到达时间等信息

## Components and Interfaces

### 新增数据结构

#### Station
```python
@dataclass
class Station:
    """公交站点"""
    station_id: int              # 站点ID
    station_name: str            # 站点名称
    waiting_passengers: List[Passenger]  # 等待队列
```

#### Passenger (增强版)
```python
@dataclass
class Passenger:
    """乘客"""
    passenger_id: int            # 乘客ID
    arrival_time: int            # 到达站点的时间(分钟)
    boarding_station: int        # 上车站点ID
    alighting_station: int       # 下车站点ID
    boarding_time: Optional[int] = None  # 实际上车时间
```

#### BusTrip (增强版)
```python
@dataclass
class BusTrip:
    """公交行程"""
    trip_id: int
    departure_time: int          # 起点站发车时间
    direction: str               # 方向
    capacity: int                # 容量
    stations: List[int]          # 经过的站点ID列表
    current_passengers: List[Passenger]  # 当前车上的乘客
    station_arrival_times: Dict[int, int]  # 各站点到达时间
    completed: bool = False      # 是否完成
```

### 修改的组件

#### DirectionState (增强版)
```python
@dataclass
class DirectionState:
    """单方向状态"""
    stations: List[Station]      # 该方向的所有站点
    buses_in_service: List[BusTrip]
    total_departures: int
    total_passengers_served: int
    total_waiting_time: float    # 所有乘客的累计等待时间
    last_departure_time: int
    
    @property
    def stranded_passengers(self) -> int:
        """服务结束时所有站点的等待乘客总数"""
        return sum(len(station.waiting_passengers) for station in self.stations)
```

## Data Models

### 站点配置

从真实数据中提取站点信息：

```python
# 208路线示例
route_208_stations = {
    'direction_0': [1, 2, 3, ..., 20],  # 上行站点ID
    'direction_1': [20, 19, 18, ..., 1]  # 下行站点ID
}
```

### 乘客数据映射

从CSV数据映射到Passenger对象：

```python
# CSV: Label, Boarding time, Boarding station, Alighting station, Arrival time
passenger = Passenger(
    passenger_id=row['Label'],
    arrival_time=row['Boarding time'],  # 实际是到达站点时间
    boarding_station=row['Boarding station'],
    alighting_station=row['Alighting station'],
    boarding_time=None  # 将在上车时设置
)
```

### 行驶时间数据

使用交通数据或估算：

```python
# 站点间行驶时间(分钟)
travel_times = {
    (station_1, station_2): 2.5,
    (station_2, station_3): 3.0,
    ...
}

# 停靠时间
dwell_time = 0.5  # 每站停靠0.5分钟
```

## Implementation Details

### 核心流程

#### 1. 初始化

```python
def __init__(self):
    # 为每个方向创建站点列表
    self.up_state.stations = [
        Station(station_id=i, station_name=f"Station_{i}", waiting_passengers=[])
        for i in range(1, num_stations + 1)
    ]
    
    # 加载乘客数据并分配到各站点
    self._distribute_passengers_to_stations()
```

#### 2. 乘客到达站点

```python
def _update_passenger_arrivals(self):
    """更新乘客到达各站点"""
    for passenger in self.arriving_passengers[self.current_time]:
        station = self._get_station(passenger.boarding_station, passenger.direction)
        station.waiting_passengers.append(passenger)
```

#### 3. 发车

```python
def _dispatch_bus(self, direction: str):
    """发车"""
    # 创建行程，包含所有站点
    stations = self.up_state.stations if direction == 'up' else self.down_state.stations
    station_ids = [s.station_id for s in stations]
    
    trip = BusTrip(
        trip_id=self.trip_counter,
        departure_time=self.current_time,
        direction=direction,
        capacity=self.bus_capacity,
        stations=station_ids,
        current_passengers=[],
        station_arrival_times={}
    )
    
    # 计算各站点到达时间
    current_time = self.current_time
    for i, station_id in enumerate(station_ids):
        if i == 0:
            trip.station_arrival_times[station_id] = current_time
        else:
            # 行驶时间 + 停靠时间
            travel_time = self._get_travel_time(station_ids[i-1], station_id)
            current_time += travel_time + self.dwell_time
            trip.station_arrival_times[station_id] = current_time
    
    # 添加到在途列表
    state.buses_in_service.append(trip)
```

#### 4. 更新在途公交车

```python
def _update_buses(self):
    """更新所有在途公交车"""
    for trip in self.up_state.buses_in_service + self.down_state.buses_in_service:
        if trip.completed:
            continue
            
        # 检查是否到达下一个站点
        next_station = self._get_next_station(trip)
        if next_station and self.current_time >= trip.station_arrival_times[next_station]:
            self._process_station_stop(trip, next_station)
```

#### 5. 站点停靠处理

```python
def _process_station_stop(self, trip: BusTrip, station_id: int):
    """处理公交车在站点的停靠"""
    station = self._get_station(station_id, trip.direction)
    
    # 1. 乘客下车
    alighting_passengers = [p for p in trip.current_passengers 
                           if p.alighting_station == station_id]
    for passenger in alighting_passengers:
        trip.current_passengers.remove(passenger)
    
    # 2. 乘客上车
    available_capacity = trip.capacity - len(trip.current_passengers)
    boarding_passengers = station.waiting_passengers[:available_capacity]
    
    for passenger in boarding_passengers:
        # 记录上车时间
        passenger.boarding_time = self.current_time
        
        # 计算等待时间
        waiting_time = passenger.boarding_time - passenger.arrival_time
        state = self._get_state_for_direction(trip.direction)
        state.total_waiting_time += waiting_time
        state.total_passengers_served += 1
        
        # 乘客上车
        trip.current_passengers.append(passenger)
        station.waiting_passengers.remove(passenger)
    
    # 3. 检查是否完成行程
    if station_id == trip.stations[-1]:
        trip.completed = True
```

### 等待时间计算

严格按照论文公式2.6：

```python
# Wₘ = Σ(k=1 to K-1) Σ(i=1 to l^k_m) (t_b^(m,i,k) - t^(m,i,k))

# 在乘客上车时计算
waiting_time = passenger.boarding_time - passenger.arrival_time
state.total_waiting_time += waiting_time

# 平均等待时间
average_waiting_time = state.total_waiting_time / state.total_passengers_served
```

### 滞留乘客计算

```python
# 服务结束时，统计所有站点的等待乘客
stranded_passengers = sum(
    len(station.waiting_passengers) 
    for station in state.stations
)
```

## Data Flow

### 完整的时间步流程

```
1. current_time += 1
2. 更新乘客到达各站点
3. 执行发车决策(如果有)
4. 更新所有在途公交车
   - 检查是否到达站点
   - 处理站点停靠(下车、上车)
   - 计算等待时间
5. 返回当前状态
```

## Compatibility

### 保持接口兼容

虽然内部实现完全重构，但保持外部接口不变：

```python
# 状态空间：10维，不变
def _get_current_state(self) -> Dict[str, Any]:
    # 返回相同格式的状态
    pass

# 动作空间：(a_up, a_down)，不变
def step(self, action: Tuple[int, int]):
    # 接受相同格式的动作
    pass

# 统计信息：相同格式，不变
def get_statistics(self) -> Dict[str, Any]:
    # 返回相同格式的统计
    pass
```

## Performance Considerations

### 优化策略

1. **预计算站点到达时间**：发车时一次性计算所有站点到达时间
2. **索引优化**：使用字典快速查找站点
3. **批量处理**：同一时刻到达的多辆车批量处理
4. **懒惰更新**：只在需要时更新公交车状态

### 复杂度分析

- **时间复杂度**：O(T × B × S)
  - T: 时间步数 (~900)
  - B: 在途公交车数 (~10)
  - S: 站点数 (~20)
  - 总计：~180,000 操作/episode

- **空间复杂度**：O(S × P + B × P)
  - S: 站点数
  - P: 乘客数 (~5000)
  - B: 公交车数

## Testing Strategy

### 单元测试

1. **站点队列测试**：验证乘客正确添加到站点队列
2. **上下车测试**：验证乘客在正确站点上下车
3. **等待时间测试**：验证等待时间计算正确
4. **行程模拟测试**：验证公交车正确经过所有站点

### 集成测试

1. **完整episode测试**：运行完整episode，验证所有指标
2. **数据一致性测试**：验证乘客数量守恒
3. **性能测试**：验证运行速度可接受

## Migration Plan

### 渐进式迁移

1. **Phase 1**：创建新的站点级别模拟器（独立文件）
2. **Phase 2**：验证新模拟器的正确性
3. **Phase 3**：替换旧模拟器
4. **Phase 4**：清理旧代码

### 回滚策略

保留旧模拟器作为备份，通过配置切换：

```python
if use_station_level_simulation:
    env = StationLevelBusEnvironment(...)
else:
    env = BidirectionalBusEnvironment(...)  # 旧版本
```

## Expected Results

实现站点级别模拟后，预期：

1. **等待时间**：更准确，应该接近论文的3.7/3.8分钟
2. **滞留乘客**：更准确，应该接近论文的0人
3. **模型真实性**：完全符合论文的模型描述
4. **可解释性**：可以分析每个站点的服务质量

## Risks and Mitigation

### 风险

1. **实现复杂度高**：站点级别模拟比当前实现复杂得多
2. **性能问题**：可能导致训练速度变慢
3. **数据问题**：真实数据可能有缺失或错误

### 缓解措施

1. **分阶段实现**：先实现核心功能，再优化
2. **性能监控**：持续监控运行速度，及时优化
3. **数据验证**：加载数据时进行完整性检查
4. **充分测试**：每个组件都有单元测试
