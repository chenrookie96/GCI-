# Requirements Document

## Introduction

根据DRL-TSBC论文的说明，公交系统的首班车和末班车时间是固定的，在这两个时间点必须进行发车。这是一个重要的运营约束，需要在站点级别模拟器中实现。

## Glossary

- **System**: 站点级别公交模拟器 (StationLevelBusEnvironment)
- **First Bus**: 首班车，在服务开始时间(service_start)发出的第一辆公交车
- **Last Bus**: 末班车，在服务结束时间(service_end)发出的最后一辆公交车
- **Fixed Departure**: 固定发车，指在特定时间点必须发车，不受DRL智能体决策影响
- **DRL Agent**: 深度强化学习智能体，负责在首末班车之间做出发车决策

## Requirements

### Requirement 1: 首班车固定发车

**User Story:** 作为公交运营系统，我希望在服务开始时间自动发出首班车，以确保服务准时开始

#### Acceptance Criteria

1. WHEN THE System resets to initial state, THE System SHALL schedule a fixed departure for both directions at service_start time
2. WHEN THE System reaches service_start time, THE System SHALL dispatch buses for both up and down directions regardless of DRL Agent action
3. THE System SHALL record the first bus departure time as service_start for both directions
4. THE System SHALL mark the first departure as "fixed" to distinguish it from DRL-controlled departures

### Requirement 2: 末班车固定发车

**User Story:** 作为公交运营系统，我希望在服务结束时间自动发出末班车，以确保所有时段都有服务覆盖

#### Acceptance Criteria

1. WHEN THE System reaches service_end time, THE System SHALL dispatch buses for both up and down directions regardless of DRL Agent action
2. THE System SHALL record the last bus departure time as service_end for both directions
3. THE System SHALL mark the last departure as "fixed" to distinguish it from DRL-controlled departures
4. WHEN THE last bus is dispatched, THE System SHALL set episode done flag to true

### Requirement 3: DRL决策时间窗口

**User Story:** 作为DRL智能体，我希望只在首末班车之间的时间窗口内做出发车决策，以避免与固定发车冲突

#### Acceptance Criteria

1. WHEN THE current_time equals service_start, THE System SHALL NOT accept DRL Agent dispatch actions
2. WHEN THE current_time equals service_end, THE System SHALL NOT accept DRL Agent dispatch actions
3. WHEN THE current_time is between service_start and service_end (exclusive), THE System SHALL accept and execute DRL Agent dispatch actions
4. THE System SHALL provide clear feedback to DRL Agent about whether its action was executed or overridden by fixed departure

### Requirement 4: 统计信息准确性

**User Story:** 作为系统分析员，我希望统计信息能够区分固定发车和DRL控制发车，以便评估算法性能

#### Acceptance Criteria

1. THE System SHALL maintain separate counters for fixed departures and DRL-controlled departures
2. THE System SHALL include fixed departure information in statistics output
3. WHEN THE System calculates total departures, THE System SHALL include both fixed and DRL-controlled departures
4. THE System SHALL provide a method to query the ratio of DRL-controlled departures to total departures

### Requirement 5: 向后兼容性

**User Story:** 作为开发者，我希望新的固定发车机制不会破坏现有的测试和训练代码

#### Acceptance Criteria

1. THE System SHALL maintain the same step() method signature
2. THE System SHALL maintain the same state representation format
3. THE System SHALL maintain the same reward calculation logic
4. WHEN THE existing test code runs, THE System SHALL pass all existing unit and integration tests
