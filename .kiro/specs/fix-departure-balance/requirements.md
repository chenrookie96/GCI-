# Requirements Document

## Introduction

修复DRL-TSBC算法中的发车次数平衡问题。当前训练结果显示上下行发车次数差异过大（上行157.9次，下行91.3次，差异66.66次），且总发车次数远超论文预期（249.2次 vs 73次）。需要调整奖励函数和约束机制，使其符合论文规范。

## Glossary

- **DRL-TSBC**: Deep Reinforcement Learning-based dynamic bus Timetable Scheduling method with Bidirectional Constraints（基于深度强化学习的双向约束动态公交时刻表排班方法）
- **发车次数平衡**: 上行和下行方向的发车次数应该相等或接近相等
- **ζ (zeta)**: 奖励函数中的发车次数差异权重参数
- **ω (omega)**: 奖励函数中的等待时间权重参数
- **约束惩罚**: 对违反双向平衡约束的额外惩罚

## Requirements

### Requirement 1: 发车次数平衡约束

**User Story:** 作为算法开发者，我希望上下行发车次数保持平衡（差异为0或接近0），以符合论文中的双向约束要求

#### Acceptance Criteria

1. WHEN 训练完成后，THE System SHALL 确保最后50轮的平均上下行发车次数差异小于1次
2. WHEN 每个episode结束时，THE System SHALL 计算并记录上下行发车次数差异
3. WHEN 发车次数差异超过阈值时，THE System SHALL 应用额外的惩罚到奖励函数中
4. THE System SHALL 在奖励函数中使用合适的ζ参数值来强化发车次数平衡约束
5. THE System SHALL 优先目标是实现发车次数差异为0次

### Requirement 2: 总发车次数控制

**User Story:** 作为算法开发者，我希望总发车次数接近论文中的预期值（约73次），而不是当前的249次

#### Acceptance Criteria

1. WHEN 训练完成后，THE System SHALL 确保最后50轮的平均总发车次数在70-76次之间
2. THE System SHALL 通过调整ω参数来控制发车频率，目标为约73次发车
3. WHEN 发车次数过多时，THE System SHALL 增加不发车动作的奖励
4. THE System SHALL 在训练过程中监控并记录每轮的总发车次数

### Requirement 3: 奖励函数优化

**User Story:** 作为算法开发者，我希望奖励函数能够正确引导智能体学习到平衡的发车策略

#### Acceptance Criteria

1. THE System SHALL 实现论文公式(2.16)和(2.17)中的奖励函数
2. WHEN 计算奖励时，THE System SHALL 使用发车次数差异的平方项来增强惩罚效果
3. THE System SHALL 支持动态调整ζ和ω参数
4. WHEN 上下行发车次数差异增大时，THE System SHALL 对差异较大的方向施加更强的惩罚

### Requirement 4: 约束参数调优

**User Story:** 作为算法开发者，我希望找到最优的约束参数组合，使算法达到论文中的性能指标

#### Acceptance Criteria

1. THE System SHALL 支持配置不同的ζ值（建议范围：0.001-0.01）
2. THE System SHALL 支持配置不同的ω值（建议范围：1/5000 - 1/500）
3. WHEN 参数变化时，THE System SHALL 记录对应的训练结果以便对比
4. THE System SHALL 提供参数调优的实验脚本

### Requirement 5: 训练监控增强

**User Story:** 作为算法开发者，我希望在训练过程中实时监控发车次数平衡情况，以便及时发现问题

#### Acceptance Criteria

1. THE System SHALL 在每个episode输出时显示上下行发车次数和差异
2. THE System SHALL 绘制发车次数差异的变化曲线
3. THE System SHALL 在训练曲线中添加发车次数平衡度指标
4. WHEN 发车次数差异持续增大时，THE System SHALL 输出警告信息
