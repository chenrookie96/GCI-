# Requirements Document

## Introduction

实现完整的站点级别公交模拟，以符合论文的标准。当前实现假设所有乘客在起点站上车，但论文使用了复杂的多站点模型，其中乘客在不同站点上下车，公交车沿途停靠多个站点。这个spec将重构环境模拟器以支持站点级别的模拟。

## Glossary

- **Station**: 公交站点，公交车沿途停靠的位置
- **Boarding Station**: 上车站点，乘客上车的站点
- **Alighting Station**: 下车站点，乘客下车的站点
- **Station-Level Simulation**: 站点级别模拟，模拟公交车在各站点的停靠、乘客上下车
- **Bus Trip**: 公交行程，从起点站到终点站的完整行程
- **Dwell Time**: 停靠时间，公交车在站点停靠的时间
- **Travel Time**: 行驶时间，公交车在两个站点之间的行驶时间
- **Passenger Waiting Time**: 乘客等待时间，从乘客到达站点到上车的时间

## Requirements

### Requirement 1

**User Story:** 作为系统开发者，我需要实现站点级别的公交车行程模拟，以便准确模拟公交车在各站点的停靠和行驶

#### Acceptance Criteria

1. WHEN THE System 创建公交行程, THE System SHALL 包含该行程经过的所有站点列表
2. WHEN 公交车行驶时, THE System SHALL 模拟公交车依次到达每个站点
3. WHEN 公交车到达站点, THE System SHALL 计算到达时间基于行驶时间和停靠时间
4. WHEN 公交车在站点停靠, THE System SHALL 处理乘客上下车
5. WHEN 计算行程总时间, THE System SHALL 包含所有站点间的行驶时间和停靠时间

### Requirement 2

**User Story:** 作为系统开发者，我需要在每个站点维护等待队列，以便准确跟踪每个站点的乘客

#### Acceptance Criteria

1. WHEN THE System 初始化环境, THE System SHALL 为每个站点创建独立的等待队列
2. WHEN 乘客到达站点, THE System SHALL 将乘客添加到对应站点的等待队列
3. WHEN 公交车到达站点, THE System SHALL 从该站点的等待队列中让乘客上车
4. WHEN 乘客上车, THE System SHALL 记录乘客的上车站点和下车站点
5. WHEN 公交车到达乘客的下车站点, THE System SHALL 让乘客下车

### Requirement 3

**User Story:** 作为系统开发者，我需要使用真实数据中的站点信息，以便准确模拟乘客行为

#### Acceptance Criteria

1. WHEN THE System 加载乘客数据, THE System SHALL 提取每个乘客的上车站点
2. WHEN THE System 加载乘客数据, THE System SHALL 提取每个乘客的下车站点
3. WHEN THE System 加载乘客数据, THE System SHALL 提取每个乘客的到达时间
4. WHEN 乘客到达时间到来, THE System SHALL 将乘客添加到对应上车站点的等待队列
5. WHEN 计算等待时间, THE System SHALL 使用乘客到达站点的时间和公交车到达该站点的时间

### Requirement 4

**User Story:** 作为系统开发者，我需要按照论文公式2.6计算等待时间，以便与论文结果一致

#### Acceptance Criteria

1. WHEN 乘客上车, THE System SHALL 计算该乘客的等待时间为上车时间减去到达站点时间
2. WHEN 累计等待时间, THE System SHALL 对所有站点的所有乘客求和
3. WHEN 计算平均等待时间, THE System SHALL 使用总等待时间除以总服务乘客数
4. WHEN 报告等待时间, THE System SHALL 以分钟为单位显示

### Requirement 5

**User Story:** 作为系统开发者，我需要正确计算滞留乘客，以便与论文定义一致

#### Acceptance Criteria

1. WHEN 公交车到达站点且已满载, THE System SHALL 无法让更多乘客上车
2. WHEN 服务结束时, THE System SHALL 统计所有站点仍在等待的乘客总数
3. WHEN 报告滞留乘客, THE System SHALL 显示服务结束时未被服务的乘客数量
4. WHEN 乘客无法上车, THE System SHALL 让乘客继续在站点等待下一班车

### Requirement 6

**User Story:** 作为系统开发者，我需要使用真实的站点间行驶时间，以便准确模拟公交车运行

#### Acceptance Criteria

1. WHEN THE System 加载交通数据, THE System SHALL 提取站点间的行驶时间
2. WHEN 公交车在两站点间行驶, THE System SHALL 使用真实的行驶时间数据
3. WHEN 真实数据不可用, THE System SHALL 使用平均行驶时间估算
4. WHEN 计算到达时间, THE System SHALL 累加所有经过站点的行驶时间和停靠时间

### Requirement 7

**User Story:** 作为系统开发者，我需要保持与现有DRL-TSBC算法的兼容性，以便无缝集成新的模拟器

#### Acceptance Criteria

1. WHEN THE System 提供状态, THE System SHALL 保持10维状态空间不变
2. WHEN THE System 接收动作, THE System SHALL 保持双向发车决策接口不变
3. WHEN THE System 计算奖励, THE System SHALL 使用相同的奖励函数
4. WHEN THE System 报告统计, THE System SHALL 提供相同格式的统计信息
5. WHEN 训练脚本调用环境, THE System SHALL 无需修改训练脚本即可使用新模拟器
