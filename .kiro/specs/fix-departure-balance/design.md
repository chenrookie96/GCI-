# Design Document

## Overview

本设计文档描述如何修复DRL-TSBC算法中的发车次数平衡问题。核心策略是通过增强奖励函数中的双向约束惩罚项，并优化约束参数，使智能体学习到平衡的发车策略。

### 问题根源分析

当前问题的根本原因：

1. **ζ参数过小**: 当前ζ=0.002，对发车次数差异的惩罚不够强
2. **线性惩罚不足**: 使用线性惩罚 `ζ * (c_up - c_down)` 无法有效约束大的差异
3. **奖励函数不对称**: 上下行奖励函数中的差异项符号相同，导致无法形成有效的平衡机制
4. **ω参数不当**: 当前ω=1/1000可能导致发车过于频繁

### 解决方案概述

1. 增强发车次数差异惩罚（使用平方项）
2. 修正奖励函数中的符号错误
3. 调整ζ和ω参数到合适范围
4. 添加全局平衡约束惩罚
5. 增强训练监控和可视化

## Architecture

### 组件关系图

```
┌─────────────────────────────────────────────────────────────┐
│                    DRL-TSBC Training System                  │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐         ┌──────────────────┐          │
│  │  Environment     │◄────────┤  DRLTSBCAgent    │          │
│  │  (Bidirectional) │         │  - Q-Network     │          │
│  │                  │         │  - Reward Func   │          │
│  │  - Up State      │         │  - Constraints   │          │
│  │  - Down State    │         └────────┬─────────┘          │
│  │  - Passengers    │                  │                     │
│  └────────┬─────────┘                  │                     │
│           │                            │                     │
│           │         ┌──────────────────▼─────┐               │
│           └────────►│  Enhanced Reward       │               │
│                     │  - Balance Penalty²    │               │
│                     │  - Adjusted ζ & ω      │               │
│                     │  - Global Constraint   │               │
│                     └────────────────────────┘               │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           Training Monitor (Enhanced)                 │   │
│  │  - Departure Balance Tracking                         │   │
│  │  - Real-time Alerts                                   │   │
│  │  - Balance Curve Visualization                        │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Components and Interfaces

### 1. Enhanced Reward Function

#### 核心修改

**当前问题（论文公式2.16和2.17）：**

```python
# 上行奖励
if a_up == 0:
    r_up = (1 - o_up) - (ω * w_up) - (β * d_up) - ζ * (c_up - c_down)
else:
    r_up = o_up - (β * d_up) + ζ * (c_up - c_down)

# 下行奖励
if a_down == 0:
    r_down = (1 - o_down) - (ω * w_down) - (β * d_down) + ζ * (c_up - c_down)
else:
    r_down = o_down - (β * d_down) - ζ * (c_up - c_down)
```

**问题分析：**
- 上行发车时：`+ ζ * (c_up - c_down)` - 当c_up > c_down时，奖励增加（错误！）
- 下行发车时：`- ζ * (c_up - c_down)` - 当c_up > c_down时，奖励减少（正确）
- 这导致上行倾向于多发车，下行倾向于少发车

**修正方案：**

```python
# 计算发车次数差异（使用平方惩罚）
departure_diff = c_up - c_down
balance_penalty = ζ * (departure_diff ** 2)

# 上行奖励（修正版）
if a_up == 0:  # 不发车
    # 如果上行已经多发车(c_up > c_down)，不发车应该得到奖励
    r_up = (1 - o_up) - (ω * w_up) - (β * d_up) - balance_penalty
else:  # 发车
    # 如果上行已经多发车(c_up > c_down)，发车应该受到惩罚
    r_up = o_up - (β * d_up) - balance_penalty
    if departure_diff > 0:  # 上行已经多了
        r_up -= ζ * abs(departure_diff)  # 额外惩罚

# 下行奖励（修正版）
if a_down == 0:  # 不发车
    # 如果下行已经少发车(c_up > c_down)，不发车应该受到惩罚
    r_down = (1 - o_down) - (ω * w_down) - (β * d_down) - balance_penalty
else:  # 发车
    # 如果下行已经少发车(c_up > c_down)，发车应该得到奖励
    r_down = o_down - (β * d_down) - balance_penalty
    if departure_diff > 0:  # 下行需要追赶
        r_down += ζ * abs(departure_diff)  # 额外奖励
```

#### 参数调整策略

| 参数 | 当前值 | 建议值 | 说明 |
|------|--------|--------|------|
| ζ (zeta) | 0.002 | 0.01 - 0.05 | 增强发车次数差异惩罚 |
| ω (omega) | 1/1000 | 1/2000 - 1/5000 | 降低发车频率 |
| β (beta) | 0.2 | 0.2 | 保持不变 |

### 2. Constraint Enforcement

#### 硬约束检查

在动作选择后添加硬约束检查：

```python
def apply_balance_constraint(self, action, up_count, down_count, threshold=5):
    """
    应用发车次数平衡硬约束
    
    如果差异超过阈值，强制调整动作
    """
    a_up, a_down = action
    diff = up_count - down_count
    
    # 如果上行多发车超过阈值
    if diff > threshold:
        if a_up == 1:  # 禁止上行发车
            a_up = 0
        if a_down == 0 and self._can_dispatch('down'):  # 强制下行发车
            a_down = 1
    
    # 如果下行多发车超过阈值
    elif diff < -threshold:
        if a_down == 1:  # 禁止下行发车
            a_down = 0
        if a_up == 0 and self._can_dispatch('up'):  # 强制上行发车
            a_up = 1
    
    return (a_up, a_down)
```

### 3. Training Monitor Enhancement

#### 新增监控指标

```python
class EnhancedTrainingMonitor(TrainingMonitor):
    def __init__(self, save_dir: str = "results"):
        super().__init__(save_dir)
        self.departure_differences = []  # 发车次数差异
        self.balance_violations = []     # 平衡约束违反次数
        
    def log_episode(self, episode, reward, loss, up_deps, down_deps, epsilon):
        super().log_episode(episode, reward, loss, up_deps, down_deps, epsilon)
        
        # 记录差异
        diff = abs(up_deps - down_deps)
        self.departure_differences.append(diff)
        
        # 检查违反
        if diff > 1:
            self.balance_violations.append(episode)
            
    def plot_balance_analysis(self):
        """绘制平衡分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 发车次数差异曲线
        axes[0, 0].plot(self.departure_differences)
        axes[0, 0].axhline(y=1, color='r', linestyle='--', label='目标阈值')
        axes[0, 0].set_title('发车次数差异变化')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('|上行 - 下行|')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 上下行发车次数对比
        axes[0, 1].scatter(self.up_departures, self.down_departures, alpha=0.5)
        axes[0, 1].plot([0, 100], [0, 100], 'r--', label='完美平衡线')
        axes[0, 1].set_title('上下行发车次数散点图')
        axes[0, 1].set_xlabel('上行发车次数')
        axes[0, 1].set_ylabel('下行发车次数')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 总发车次数分布
        total_deps = [u + d for u, d in zip(self.up_departures, self.down_departures)]
        axes[1, 0].hist(total_deps, bins=30, edgecolor='black')
        axes[1, 0].axvline(x=73, color='r', linestyle='--', label='目标值(73)')
        axes[1, 0].set_title('总发车次数分布')
        axes[1, 0].set_xlabel('总发车次数')
        axes[1, 0].set_ylabel('频次')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 平衡度随时间变化
        window = 50
        rolling_balance = []
        for i in range(len(self.departure_differences)):
            if i >= window:
                rolling_balance.append(
                    np.mean(self.departure_differences[i-window:i])
                )
        axes[1, 1].plot(rolling_balance)
        axes[1, 1].axhline(y=1, color='r', linestyle='--', label='目标阈值')
        axes[1, 1].set_title(f'平衡度滚动平均(窗口={window})')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('平均差异')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'balance_analysis.png', dpi=300)
        plt.close()
```

## Data Models

### RewardFunctionConfig

```python
@dataclass
class RewardFunctionConfig:
    """奖励函数配置"""
    zeta: float = 0.02          # 发车次数差异权重（增强）
    omega: float = 1.0/3000     # 等待时间权重（降低）
    beta: float = 0.2           # 滞留乘客惩罚权重
    use_squared_penalty: bool = True  # 使用平方惩罚
    balance_threshold: int = 5  # 硬约束阈值
```

### BalanceMetrics

```python
@dataclass
class BalanceMetrics:
    """平衡度指标"""
    episode: int
    up_departures: int
    down_departures: int
    difference: int
    total_departures: int
    balance_score: float  # 1 - (diff / total)
    meets_target: bool    # diff <= 1
```

## Error Handling

### 训练异常处理

1. **发车次数爆炸**: 如果单个episode发车次数超过200次，终止该episode并记录警告
2. **平衡度持续恶化**: 如果连续50个episode平衡度都>10，输出警告并建议调整参数
3. **奖励崩溃**: 如果奖励持续为负且绝对值很大，检查参数设置

```python
def check_training_health(self, monitor, episode):
    """检查训练健康度"""
    if episode < 50:
        return
    
    recent_diffs = monitor.departure_differences[-50:]
    avg_diff = np.mean(recent_diffs)
    
    if avg_diff > 10:
        logger.warning(f"Episode {episode}: 平衡度持续恶化 (平均差异={avg_diff:.1f})")
        logger.warning("建议: 增大ζ参数或降低ω参数")
    
    recent_total = [u + d for u, d in zip(
        monitor.up_departures[-50:],
        monitor.down_departures[-50:]
    )]
    avg_total = np.mean(recent_total)
    
    if avg_total > 150:
        logger.warning(f"Episode {episode}: 发车次数过多 (平均={avg_total:.1f})")
        logger.warning("建议: 降低ω参数")
    elif avg_total < 50:
        logger.warning(f"Episode {episode}: 发车次数过少 (平均={avg_total:.1f})")
        logger.warning("建议: 增大ω参数")
```

## Testing Strategy

### 单元测试

1. **奖励函数测试**: 验证修正后的奖励函数在不同场景下的行为
2. **约束测试**: 验证硬约束能够正确限制动作
3. **参数敏感性测试**: 测试不同ζ和ω值对结果的影响

### 集成测试

1. **短期训练测试**: 运行50个episodes，验证基本功能
2. **平衡度测试**: 验证发车次数差异是否收敛到目标范围
3. **总发车次数测试**: 验证总发车次数是否接近73次

### 参数调优实验

设计参数网格搜索实验：

```python
param_grid = {
    'zeta': [0.01, 0.02, 0.05],
    'omega': [1/2000, 1/3000, 1/5000]
}

# 对每组参数运行3次实验，取平均结果
```

## Implementation Notes

### 实现优先级

1. **高优先级**: 修正奖励函数（核心问题）
2. **高优先级**: 调整ζ和ω参数
3. **中优先级**: 添加硬约束检查
4. **中优先级**: 增强训练监控
5. **低优先级**: 参数调优实验脚本

### 向后兼容性

- 保留原始奖励函数作为`calculate_reward_v1`
- 新奖励函数命名为`calculate_reward_v2`
- 通过配置参数选择使用哪个版本

### 性能考虑

- 平方惩罚计算开销很小，不影响训练速度
- 硬约束检查在每步执行，但逻辑简单，开销可忽略
- 增强监控会增加少量内存使用，但在可接受范围内
