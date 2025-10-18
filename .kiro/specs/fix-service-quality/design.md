# Design Document

## Overview

本设计文档详细说明如何修复DRL-TSBC系统中的服务质量问题。通过修正环境模拟器中的发车逻辑、滞留乘客计算和等待时间计算，使系统达到论文表2-3的性能水平。

## Architecture

修复将集中在 `BidirectionalBusEnvironment` 类的 `_dispatch_bus` 方法中，这是处理发车、乘客上车和服务质量指标计算的核心方法。

### 当前问题分析

#### 问题1：滞留乘客计算错误

**当前代码（第145-146行）：**
```python
state.waiting_passengers = max(0, state.waiting_passengers - passengers_to_board)
state.stranded_passengers += max(0, state.waiting_passengers - passengers_to_board)
```

**问题：**
- 第145行：先更新了 `waiting_passengers`，减去了已上车的乘客
- 第146行：然后用更新后的 `waiting_passengers` 计算滞留乘客
- 结果：所有未上车的乘客都被计为滞留乘客，即使他们只是在等待下一班车

**正确逻辑：**
滞留乘客应该是**超出车辆容量无法上车的乘客**，而不是所有剩余等待的乘客。

#### 问题2：等待时间计算逻辑

**当前代码（第137-141行）：**
```python
if passengers_to_board > 0 and state.last_departure_time >= 0:
    interval = self.current_time - state.last_departure_time
    avg_waiting = interval / 2.0
    state.total_waiting_time += avg_waiting * passengers_to_board
```

**问题：**
- 使用 `state.last_departure_time >= 0` 作为条件，但初始值是 `-999`
- 首次发车时，`interval` 会是一个巨大的负数或正数
- 导致等待时间计算不准确

**正确逻辑：**
- 首次发车时不应计算等待时间（因为没有"上一次发车"）
- 应该检查 `last_departure_time` 是否为有效值（例如 > 0）

#### 问题3：乘客队列更新顺序

**当前逻辑顺序：**
1. 计算上车乘客数
2. 计算等待时间
3. 更新等待队列（减去已上车）
4. 计算滞留乘客（使用已更新的队列）

**正确顺序应该是：**
1. 计算上车乘客数
2. 计算滞留乘客（使用原始队列）
3. 更新等待队列（减去已上车）
4. 计算等待时间

## Components and Interfaces

### 修改的组件

#### BidirectionalBusEnvironment._dispatch_bus()

**输入：**
- `direction: str` - 发车方向（'up' 或 'down'）

**输出：**
- 无（更新内部状态）

**修改的状态变量：**
- `state.waiting_passengers` - 等待乘客数量
- `state.stranded_passengers` - 滞留乘客数量（累计）
- `state.total_waiting_time` - 总等待时间（累计）
- `state.total_passengers_served` - 总服务乘客数（累计）
- `state.total_departures` - 总发车次数（累计）
- `state.last_departure_time` - 上次发车时间

## Data Models

### DirectionState

保持不变，但明确各字段的语义：

```python
@dataclass
class DirectionState:
    buses_in_service: List[BidirectionalBusTrip]  # 在途车辆列表
    waiting_passengers: int                        # 当前等待乘客数
    stranded_passengers: int                       # 累计滞留乘客数（超出容量无法上车）
    total_departures: int                          # 累计发车次数
    total_passengers_served: int                   # 累计服务乘客数
    total_waiting_time: float                      # 累计等待时间（分钟）
    last_departure_time: int                       # 上次发车时间（分钟）
```

## Error Handling

### 边界情况处理

1. **首次发车**
   - 条件：`last_departure_time < 0` 或 `last_departure_time == -999`
   - 处理：不计算等待时间，直接发车

2. **无等待乘客**
   - 条件：`waiting_passengers == 0`
   - 处理：发空车，所有指标为0

3. **等待乘客超出容量**
   - 条件：`waiting_passengers > bus_capacity`
   - 处理：
     - 上车乘客 = `bus_capacity`
     - 滞留乘客 = `waiting_passengers - bus_capacity`
     - 剩余等待 = 0（所有未上车的都是滞留）

4. **等待乘客少于容量**
   - 条件：`waiting_passengers <= bus_capacity`
   - 处理：
     - 上车乘客 = `waiting_passengers`
     - 滞留乘客 = 0
     - 剩余等待 = 0

## Testing Strategy

### 单元测试

创建测试脚本验证修复后的逻辑：

1. **测试滞留乘客计算**
   - 场景1：等待50人，容量48人 → 滞留2人
   - 场景2：等待30人，容量48人 → 滞留0人
   - 场景3：等待0人，容量48人 → 滞留0人

2. **测试等待时间计算**
   - 场景1：首次发车 → 等待时间不增加
   - 场景2：间隔10分钟发车，30人上车 → 增加 5*30=150 分钟
   - 场景3：间隔6分钟发车，20人上车 → 增加 3*20=60 分钟

3. **测试完整episode**
   - 运行完整训练episode
   - 验证最终指标：
     - 等待时间：3.5-4.0分钟
     - 滞留乘客：0或接近0
     - 发车次数：70-76次，上下行差异≤1

### 集成测试

运行完整训练流程（100 episodes）并验证：
- 平均等待时间达到论文水平
- 滞留乘客数量为0或极少
- 发车平衡保持良好

## Implementation Details

### 修复后的 _dispatch_bus 方法伪代码

```python
def _dispatch_bus(self, direction: str):
    # 1. 创建新行程
    new_trip = create_trip(current_time, direction)
    
    # 2. 计算上车乘客数
    passengers_to_board = min(waiting_passengers, bus_capacity)
    
    # 3. 计算滞留乘客（使用原始等待队列）
    stranded_this_time = max(0, waiting_passengers - bus_capacity)
    stranded_passengers += stranded_this_time
    
    # 4. 更新等待队列
    waiting_passengers -= passengers_to_board
    
    # 5. 计算等待时间（仅当非首次发车时）
    if passengers_to_board > 0 and last_departure_time > 0:
        interval = current_time - last_departure_time
        avg_waiting = interval / 2.0
        total_waiting_time += avg_waiting * passengers_to_board
    
    # 6. 更新统计信息
    total_passengers_served += passengers_to_board
    total_departures += 1
    last_departure_time = current_time
    buses_in_service.append(new_trip)
```

### 关键修改点

1. **第3步**：在更新等待队列之前计算滞留乘客
2. **第4步**：只减去已上车的乘客，不使用 `max(0, ...)`
3. **第5步**：使用 `last_departure_time > 0` 而不是 `>= 0`

## Performance Expectations

修复后的预期性能（基于论文表2-3）：

| 指标 | 目标值 | 当前值 | 改进 |
|------|--------|--------|------|
| 上行发车次数 | 73 | 70-71 | ✅ 已达标 |
| 下行发车次数 | 73 | 70-71 | ✅ 已达标 |
| 上行平均等待(分钟) | 3.7 | 6.6 | ❌ 需修复 |
| 下行平均等待(分钟) | 3.8 | 6.9 | ❌ 需修复 |
| 上行滞留乘客 | 0 | 7613 | ❌ 需修复 |
| 下行滞留乘客 | 0 | 2140 | ❌ 需修复 |

修复后预期：
- 等待时间：3.5-4.0分钟（接近论文的3.7/3.8）
- 滞留乘客：0-10人（接近论文的0）
- 发车平衡：保持当前水平（差异≤1）
