# Design Document: Fixed First and Last Bus Departures

## Overview

本设计实现DRL-TSBC论文中描述的首末班车固定发车机制。系统将在服务开始和结束时间自动发车，DRL智能体只在这两个时间点之间做出发车决策。

## Architecture

### 核心设计原则

1. **最小侵入性**: 在现有StationLevelBusEnvironment基础上添加固定发车逻辑，不破坏现有结构
2. **清晰分离**: 固定发车和DRL控制发车逻辑分离，便于维护和测试
3. **透明性**: 通过统计信息清楚展示哪些发车是固定的，哪些是DRL控制的

### 修改点

1. **step()方法**: 添加固定发车检查逻辑
2. **DirectionState**: 添加固定发车计数器
3. **统计信息**: 扩展统计信息包含固定发车数据

## Components and Interfaces

### 1. DirectionState扩展

```python
@dataclass
class DirectionState:
    """单方向状态 - 添加固定发车追踪"""
    direction: str
    stations: List[Station]
    buses_in_service: List[BusTrip] = field(default_factory=list)
    total_departures: int = 0
    fixed_departures: int = 0  # 新增：固定发车次数
    drl_departures: int = 0    # 新增：DRL控制发车次数
    total_passengers_served: int = 0
    total_waiting_time: float = 0.0
    last_departure_time: int = -999
    first_bus_dispatched: bool = False  # 新增：首班车是否已发
    last_bus_dispatched: bool = False   # 新增：末班车是否已发
```

### 2. step()方法修改

```python
def step(self, action: Tuple[int, int]) -> Tuple[Dict, float, bool]:
    """
    执行一步仿真
    
    修改逻辑：
    1. 检查是否需要固定发车（首班车或末班车）
    2. 如果是固定发车时间，忽略DRL action，强制发车
    3. 否则，执行DRL action
    4. 返回状态时标记是否发生了固定发车
    """
    # 1. 检查固定发车
    fixed_dispatch_up, fixed_dispatch_down = self._check_fixed_departures()
    
    # 2. 确定最终动作
    if fixed_dispatch_up or fixed_dispatch_down:
        # 固定发车时间，覆盖DRL动作
        final_action = (
            1 if fixed_dispatch_up else action[0],
            1 if fixed_dispatch_down else action[1]
        )
        action_overridden = True
    else:
        # 正常DRL控制时间
        final_action = action
        action_overridden = False
    
    # 3. 执行发车
    if final_action[0] == 1:
        self._dispatch_bus('up', is_fixed=fixed_dispatch_up)
    if final_action[1] == 1:
        self._dispatch_bus('down', is_fixed=fixed_dispatch_down)
    
    # 4. 更新环境
    self._update_passenger_arrivals()
    self._update_buses()
    
    # 5. 计算奖励
    reward = self._calculate_reward()
    
    # 6. 检查是否结束
    done = self.is_done()
    
    # 7. 获取状态（包含固定发车标记）
    state = self._get_current_state()
    state['action_overridden'] = action_overridden
    state['fixed_dispatch_up'] = fixed_dispatch_up
    state['fixed_dispatch_down'] = fixed_dispatch_down
    
    # 8. 时间前进
    self.current_time += 1
    
    return state, reward, done
```

### 3. 固定发车检查逻辑

```python
def _check_fixed_departures(self) -> Tuple[bool, bool]:
    """
    检查当前时间是否需要固定发车
    
    Returns:
        (up_fixed, down_fixed): 两个方向是否需要固定发车
    """
    up_fixed = False
    down_fixed = False
    
    # 首班车检查
    if self.current_time == self.service_start:
        if not self.up_state.first_bus_dispatched:
            up_fixed = True
        if not self.down_state.first_bus_dispatched:
            down_fixed = True
    
    # 末班车检查
    elif self.current_time == self.service_end:
        if not self.up_state.last_bus_dispatched:
            up_fixed = True
        if not self.down_state.last_bus_dispatched:
            down_fixed = True
    
    return up_fixed, down_fixed
```

### 4. 发车方法修改

```python
def _dispatch_bus(self, direction: str, is_fixed: bool = False):
    """
    发车
    
    Args:
        direction: 方向 ('up' or 'down')
        is_fixed: 是否为固定发车（首班车或末班车）
    """
    state = self.up_state if direction == 'up' else self.down_state
    
    # 创建行程
    self.trip_counter += 1
    stations = [s.station_id for s in state.stations]
    trip = BusTrip(
        trip_id=self.trip_counter,
        departure_time=self.current_time,
        direction=direction,
        capacity=self.bus_capacity,
        stations=stations
    )
    
    # 计算到达时间
    self._calculate_trip_times(trip)
    
    # 添加到在途列表
    state.buses_in_service.append(trip)
    
    # 更新统计
    state.total_departures += 1
    if is_fixed:
        state.fixed_departures += 1
        # 标记首末班车
        if self.current_time == self.service_start:
            state.first_bus_dispatched = True
        elif self.current_time == self.service_end:
            state.last_bus_dispatched = True
    else:
        state.drl_departures += 1
    
    state.last_departure_time = self.current_time
    
    if self.enable_logging:
        dispatch_type = "FIXED" if is_fixed else "DRL"
        logger.info(f"[{dispatch_type}] 发车 - 方向:{direction}, 时间:{self.current_time}, "
                   f"行程ID:{trip.trip_id}")
```

### 5. 统计信息扩展

```python
def get_statistics(self) -> Dict[str, Any]:
    """获取统计信息 - 包含固定发车信息"""
    return {
        'up_statistics': {
            'total_departures': self.up_state.total_departures,
            'fixed_departures': self.up_state.fixed_departures,
            'drl_departures': self.up_state.drl_departures,
            'drl_control_ratio': (self.up_state.drl_departures / self.up_state.total_departures 
                                 if self.up_state.total_departures > 0 else 0),
            'total_passengers_served': self.up_state.total_passengers_served,
            'total_waiting_time': self.up_state.total_waiting_time,
            'stranded_passengers': self.up_state.stranded_passengers,
            'first_bus_time': self.service_start if self.up_state.first_bus_dispatched else None,
            'last_bus_time': self.service_end if self.up_state.last_bus_dispatched else None,
        },
        'down_statistics': {
            'total_departures': self.down_state.total_departures,
            'fixed_departures': self.down_state.fixed_departures,
            'drl_departures': self.down_state.drl_departures,
            'drl_control_ratio': (self.down_state.drl_departures / self.down_state.total_departures 
                                 if self.down_state.total_departures > 0 else 0),
            'total_passengers_served': self.down_state.total_passengers_served,
            'total_waiting_time': self.down_state.total_waiting_time,
            'stranded_passengers': self.down_state.stranded_passengers,
            'first_bus_time': self.service_start if self.down_state.first_bus_dispatched else None,
            'last_bus_time': self.service_end if self.down_state.last_bus_dispatched else None,
        },
        'bidirectional_constraints': {
            'departure_count_difference': abs(self.up_state.total_departures - self.down_state.total_departures),
        }
    }
```

## Data Models

### 状态表示扩展

当前状态字典添加以下字段：

```python
{
    # 现有字段...
    'up_waiting_passengers': int,
    'down_waiting_passengers': int,
    # ... 其他现有字段 ...
    
    # 新增字段
    'action_overridden': bool,      # DRL动作是否被固定发车覆盖
    'fixed_dispatch_up': bool,      # 上行是否发生固定发车
    'fixed_dispatch_down': bool,    # 下行是否发生固定发车
    'up_first_bus_dispatched': bool,  # 上行首班车是否已发
    'down_first_bus_dispatched': bool, # 下行首班车是否已发
}
```

## Error Handling

### 边界情况处理

1. **service_start == service_end**: 
   - 只发一次车（既是首班车也是末班车）
   - episode立即结束

2. **service_end - service_start == 1**:
   - 第一分钟发首班车
   - 第二分钟发末班车
   - DRL没有决策空间

3. **重复发车检查**:
   - 使用first_bus_dispatched和last_bus_dispatched标志防止重复发车
   - 即使step()被多次调用，固定发车也只执行一次

## Testing Strategy

### 单元测试

1. **test_first_bus_dispatch**: 测试首班车在service_start时自动发车
2. **test_last_bus_dispatch**: 测试末班车在service_end时自动发车
3. **test_action_override**: 测试DRL动作在固定发车时间被覆盖
4. **test_drl_control_window**: 测试DRL在首末班车之间正常控制
5. **test_statistics_accuracy**: 测试统计信息正确区分固定和DRL发车

### 集成测试

1. **test_full_episode_with_fixed_buses**: 测试完整episode包含固定发车
2. **test_drl_agent_integration**: 测试DRL智能体与固定发车机制的集成
3. **test_backward_compatibility**: 测试现有代码的向后兼容性

### 边界测试

1. **test_single_minute_service**: 测试service_start == service_end情况
2. **test_two_minute_service**: 测试最小DRL决策窗口
3. **test_no_drl_actions**: 测试只有固定发车的情况

## Implementation Notes

### 实现顺序

1. 扩展DirectionState数据结构
2. 实现_check_fixed_departures()方法
3. 修改_dispatch_bus()方法支持is_fixed参数
4. 修改step()方法集成固定发车逻辑
5. 扩展get_statistics()方法
6. 更新reset()方法初始化新字段
7. 编写单元测试
8. 编写集成测试
9. 验证向后兼容性

### 注意事项

1. **时间点精确性**: 固定发车必须在精确的时间点触发
2. **状态一致性**: 确保固定发车后状态更新与DRL发车一致
3. **日志清晰性**: 日志中明确标记固定发车和DRL发车
4. **测试覆盖率**: 确保所有边界情况都有测试覆盖
