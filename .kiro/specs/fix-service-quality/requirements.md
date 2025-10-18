# Requirements Document

## Introduction

修复DRL-TSBC算法的服务质量问题，使其达到论文表2-3的水平。当前实现虽然成功平衡了上下行发车次数（70-71次，接近论文的73次），但存在严重的服务质量问题：等待时间过长（6.6/6.9分钟 vs 论文的3.7/3.8分钟）和大量滞留乘客（上行7613人，下行2140人 vs 论文的0人）。

## Glossary

- **System**: DRL-TSBC双向公交调度系统
- **Stranded Passengers**: 滞留乘客，指未能上车的乘客
- **Waiting Time**: 平均等待时间，单位为分钟
- **Departure Balance**: 发车平衡，指上下行发车次数的差异
- **Service Quality**: 服务质量，包括等待时间和滞留乘客数量
- **Dispatch Logic**: 发车逻辑，控制公交车如何接载乘客

## Requirements

### Requirement 1

**User Story:** 作为系统开发者，我需要修复滞留乘客计算逻辑，以便准确统计未能上车的乘客数量

#### Acceptance Criteria

1. WHEN THE System 执行发车操作, THE System SHALL 仅将超出车辆容量的等待乘客计入滞留乘客
2. WHEN 车辆容量足够, THE System SHALL 将滞留乘客数量设置为0
3. WHEN 计算滞留乘客, THE System SHALL 使用公式: max(0, waiting_passengers - bus_capacity)
4. WHEN 乘客上车后, THE System SHALL 从等待队列中减去已上车的乘客数量

### Requirement 2

**User Story:** 作为系统开发者，我需要修复等待时间计算逻辑，以便准确反映乘客的实际等待时间

#### Acceptance Criteria

1. WHEN 计算等待时间, THE System SHALL 使用发车间隔的一半作为平均等待时间
2. WHEN 首次发车时, THE System SHALL 不计算等待时间
3. WHEN 累计等待时间, THE System SHALL 将平均等待时间乘以上车乘客数量
4. WHEN 计算平均等待时间指标, THE System SHALL 使用总等待时间除以总服务乘客数

### Requirement 3

**User Story:** 作为系统开发者，我需要验证修复后的系统性能，以便确保达到论文水平

#### Acceptance Criteria

1. WHEN 训练完成后, THE System SHALL 报告上行平均等待时间在3.5-4.0分钟范围内
2. WHEN 训练完成后, THE System SHALL 报告下行平均等待时间在3.5-4.0分钟范围内
3. WHEN 训练完成后, THE System SHALL 报告上行滞留乘客数量为0或接近0
4. WHEN 训练完成后, THE System SHALL 报告下行滞留乘客数量为0或接近0
5. WHEN 训练完成后, THE System SHALL 保持上下行发车次数差异小于等于1

### Requirement 4

**User Story:** 作为系统开发者，我需要诊断和分析当前问题的根本原因，以便制定正确的修复方案

#### Acceptance Criteria

1. WHEN 分析代码, THE System SHALL 识别出滞留乘客计算中的逻辑错误
2. WHEN 分析代码, THE System SHALL 识别出等待时间计算中的逻辑错误
3. WHEN 分析代码, THE System SHALL 识别出乘客上车逻辑中的错误
4. WHEN 生成诊断报告, THE System SHALL 列出所有发现的问题及其影响
