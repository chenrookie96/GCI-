# 🚀 DRL-TSBC 快速开始指南

## 📋 我做了什么

我为您实现了完整的**DRL-TSBC双向动态公交时刻表排班算法**，包括：

### ✅ 已完成的工作

1. **核心算法实现** (`src/algorithms/drl_tsbc.py`)
   - DQN神经网络结构
   - 双向约束处理逻辑
   - 经验回放机制
   - ε-贪婪探索策略

2. **仿真环境** (`src/environment/bidirectional_bus_simulator.py`)
   - 双向公交线路仿真
   - 乘客流量模拟
   - 奖励函数计算
   - 状态空间管理

3. **数据处理** (`src/data_processing/passenger_data_loader.py`)
   - 真实数据加载
   - 格式转换
   - 流量矩阵生成

4. **训练脚本** (`experiments/train_drl_tsbc.py`)
   - 完整训练流程
   - 结果可视化
   - 模型保存

## 🎯 各文件的具体作用

### 1. 核心算法 - `drl_tsbc.py`
```python
# 这个文件是整个项目的核心，包含：
class DRLTSBCAgent:
    # 智能体：负责学习最优的发车策略
    def select_action()      # 选择发车动作
    def train()             # 训练神经网络
    def apply_constraints() # 应用发车间隔约束
```

**作用**: 实现DQN算法，学习在每个时间点是否发车的最优策略

### 2. 仿真环境 - `bidirectional_bus_simulator.py`
```python
class BidirectionalBusEnvironment:
    def step(action)        # 执行发车动作，返回奖励
    def get_state()         # 获取当前系统状态
    def calculate_reward()  # 计算奖励值
```

**作用**: 模拟真实的双向公交系统，为算法提供训练环境

### 3. 数据处理 - `passenger_data_loader.py`
```python
class PassengerDataLoader:
    def load_passenger_data()           # 加载乘客数据
    def generate_passenger_flow_matrix() # 生成流量矩阵
```

**作用**: 处理真实的乘客刷卡数据，转换为算法可用的格式

### 4. 训练脚本 - `train_drl_tsbc.py`
```python
def main():
    # 1. 加载数据
    # 2. 创建环境和智能体
    # 3. 训练循环
    # 4. 保存结果
```

**作用**: 协调所有组件，执行完整的训练过程

## 🔗 文件间的关系

```
训练脚本 (train_drl_tsbc.py)
    ↓
    创建智能体 (DRLTSBCAgent)
    ↓
    创建环境 (BidirectionalBusEnvironment)
    ↓
    加载数据 (PassengerDataLoader)
    ↓
    训练循环：
    - 智能体选择动作
    - 环境执行动作并返回奖励
    - 智能体学习和更新
    ↓
    保存模型和结果
```

## 🛠️ 复现操作步骤

### 第一步：环境准备
```bash
# 1. 确保Python环境 (推荐Python 3.8+)
python --version

# 2. 安装依赖包
pip install torch numpy matplotlib pandas scikit-learn seaborn tqdm jupyter

# 或者使用requirements.txt
pip install -r requirements.txt
```

### 第二步：数据准备
```bash
# 1. 创建data目录（如果不存在）
mkdir -p data

# 2. 将您的211、683线数据放入data目录
# 数据文件应该包含以下列：
# - Label: 乘客ID
# - Boarding time: 上车时间(分钟，如391表示6:31)
# - Boarding station: 上车站点
# - Alighting station: 下车站点  
# - Arrival time: 到站时间(分钟)
```

### 第三步：运行训练
```bash
# 基础训练（使用默认参数）
python experiments/train_drl_tsbc.py

# 自定义参数训练
python experiments/train_drl_tsbc.py --episodes 1000 --learning_rate 0.001
```

### 第四步：查看结果
训练完成后，检查以下文件：
```
results/
├── drl_tsbc_model.pth              # 训练好的模型
├── training_history.json           # 训练历史数据
├── drl_tsbc_training_curves.png    # 训练曲线图
└── performance_metrics.json        # 性能指标
```

## 📊 预期输出

### 训练过程输出
```
Episode 1/1000, Reward: -1250.5, Epsilon: 0.995, Up: 45, Down: 45
Episode 2/1000, Reward: -1180.2, Epsilon: 0.990, Up: 47, Down: 47
...
Episode 1000/1000, Reward: -850.3, Epsilon: 0.010, Up: 52, Down: 52

训练完成！
模型已保存到: results/drl_tsbc_model.pth
```

### 结果文件
1. **训练曲线图**: 显示奖励、探索率、发车次数的变化
2. **性能指标**: 平均等待时间、运力利用率等
3. **训练好的模型**: 可用于实际调度的神经网络

## ⚙️ 关键参数说明

### 算法参数（可在代码中调整）
```python
# 学习参数
learning_rate = 0.001    # 学习率，控制学习速度
gamma = 0.95            # 折扣因子，控制未来奖励权重
epsilon = 1.0           # 探索率，控制探索vs利用平衡

# 约束参数
tmin = 3               # 最小发车间隔(分钟)
tmax = 15              # 最大发车间隔(分钟)

# 网络结构
hidden_dims = [128, 256, 128]  # 隐藏层维度
```

### 环境参数
```python
service_start = 360     # 服务开始时间(6:00 AM)
service_end = 1320      # 服务结束时间(22:00 PM)
num_stations = 37       # 站点数量（根据实际线路调整）
```

## 🔧 常见问题解决

### 问题1：数据格式不匹配
**解决**: 检查CSV文件列名是否与代码期望一致

### 问题2：内存不足
**解决**: 减少`buffer_size`或`batch_size`

### 问题3：训练收敛慢
**解决**: 调整学习率或增加训练轮数

### 问题4：GPU不可用
**解决**: 代码会自动使用CPU，但训练会较慢

## 📈 如何验证结果

1. **收敛性**: 查看奖励曲线是否趋于稳定
2. **双向平衡**: 检查上下行发车次数是否相等
3. **约束满足**: 验证发车间隔是否在[Tmin, Tmax]范围内
4. **性能提升**: 与基准方法（如人工排班）对比

## 🎯 下一步扩展

1. **实时应用**: 将训练好的模型用于实时调度
2. **多线路**: 扩展到多条公交线路
3. **参数优化**: 根据实际效果调整超参数
4. **可视化**: 添加更丰富的结果展示

---

**总结**: 我为您提供了一个完整的DRL-TSBC算法实现，包括数据处理、算法核心、仿真环境和训练脚本。您只需要提供211、683线的数据，就可以直接运行训练，复现论文中的实验结果。
