# DRL-TSBC 使用指南

## 目录
1. [环境准备](#环境准备)
2. [快速开始](#快速开始)
3. [详细使用说明](#详细使用说明)
4. [实验复现](#实验复现)
5. [常见问题](#常见问题)

---

## 环境准备

### 1. 系统要求
- Python 3.8+
- CUDA 11.8 (GPU训练)
- 至少8GB RAM
- 至少2GB GPU显存（推荐4GB+）

### 2. 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 3. 验证安装

```bash
python tests/test_modules.py
```

如果所有测试通过，说明环境配置成功。

---

## 快速开始

### 最简单的方式

```bash
python quick_start.py
```

这将自动完成：
1. 训练208线路上行模型
2. 评估上行模型
3. 训练208线路下行模型
4. 评估下行模型
5. 生成所有图表和结果

预计时间：30-60分钟（取决于GPU性能）

---

## 详细使用说明

### 1. 训练单个模型

#### 使用命令行参数

```bash
# 基本用法
python main.py train --route 208 --direction 0 --episodes 50

# 指定设备
python main.py train --route 208 --direction 0 --device cuda

# 修改训练轮数
python main.py train --route 208 --direction 0 --episodes 100

# 设置随机种子
python main.py train --route 208 --direction 0 --seed 42
```

#### 使用配置文件

```bash
# 使用预定义配置
python main.py train --config configs/route_208_dir_0.json

# 修改配置文件后训练
# 1. 编辑 configs/route_208_dir_0.json
# 2. 运行训练
python main.py train --config configs/route_208_dir_0.json
```

### 2. 评估模型

```bash
# 基本评估
python main.py evaluate --model results/models/route_208_dir_0.pth --route 208 --direction 0

# 不保存时刻表
python main.py evaluate --model results/models/route_208_dir_0.pth --route 208 --direction 0 --no-save
```

### 3. 批量训练

```bash
# 训练所有线路（208上下行 + 211上下行）
python experiments/train_all.py --episodes 50

# 使用CPU训练
python experiments/train_all.py --episodes 50 --device cpu
```

### 4. 生成对比表格

```bash
# 生成论文中的表2-3和表2-4
python experiments/compare_methods.py
```

---

## 实验复现

### 复现论文结果

#### 1. 训练基准模型

```bash
# 208线路
python main.py train --route 208 --direction 0 --episodes 50
python main.py train --route 208 --direction 1 --episodes 50

# 211线路
python main.py train --route 211 --direction 0 --episodes 50
python main.py train --route 211 --direction 1 --episodes 50
```

#### 2. 评估所有模型

```bash
python main.py evaluate --model results/models/route_208_dir_0.pth --route 208 --direction 0
python main.py evaluate --model results/models/route_208_dir_1.pth --route 208 --direction 1
python main.py evaluate --model results/models/route_211_dir_0.pth --route 211 --direction 0
python main.py evaluate --model results/models/route_211_dir_1.pth --route 211 --direction 1
```

#### 3. 生成所有图表

训练和评估过程会自动生成以下图表：
- 训练收敛曲线（图2-11）
- 损失曲线
- 时刻表热力图

#### 4. 生成对比表格

```bash
python experiments/compare_methods.py
```

这将生成：
- 表2-3：性能对比表
- 表2-4：ω参数敏感性分析表

---

## 高级功能

### 1. 修改超参数

编辑配置文件 `configs/route_208_dir_0.json`：

```json
{
  "route_config": {
    "route_id": 208,
    "direction": 0,
    "omega": 0.001,  // 修改ω参数
    ...
  },
  "dqn_config": {
    "learning_rate": 0.001,  // 修改学习率
    "batch_size": 64,        // 修改批次大小
    "num_episodes": 50,      // 修改训练轮数
    ...
  }
}
```

### 2. 参数敏感性分析

测试不同的ω值：

```python
# 创建自定义脚本
from utils.config import ConfigManager, ExperimentConfig
from experiments.train import train

omega_values = [1/500, 1/1000, 1/2000, 1/3000, 1/4000]

for omega in omega_values:
    route_config = ConfigManager.get_route_208_config(0)
    route_config.omega = omega
    
    dqn_config = ConfigManager.get_default_dqn_config()
    config = ExperimentConfig(route_config, dqn_config)
    
    train(config, save_path=f'results/models/route_208_omega_{int(1/omega)}.pth')
```

### 3. 自定义可视化

```python
from visualization.visualizer import Visualizer
import json

vis = Visualizer()

# 加载训练数据
with open('results/models/route_208_dir_0_training_data.json', 'r') as f:
    data = json.load(f)

# 绘制自定义图表
vis.plot_training_convergence(
    episode_rewards=data['episode_rewards'],
    dispatch_counts=data['episode_dispatch_counts'],
    title='自定义标题',
    filename='custom_plot.png'
)
```

---

## 输出文件说明

### 训练输出

训练完成后会生成以下文件：

```
results/
├── models/
│   ├── route_208_dir_0.pth              # 模型权重
│   ├── route_208_dir_0_training_data.json  # 训练数据
│   ├── route_208_dir_0_schedule.json    # 生成的时刻表
│   └── route_208_dir_0_evaluation.json  # 评估结果
├── figures/
│   ├── convergence_route_208_dir_0.png  # 收敛曲线
│   ├── loss_route_208_dir_0.png         # 损失曲线
│   └── schedule_heatmap_route_208_dir_0.png  # 时刻表热力图
└── tables/
    ├── comparison_route_208.csv         # 性能对比表
    └── omega_sensitivity.csv            # 参数敏感性表
```

### 文件格式

#### 时刻表文件 (JSON)
```json
{
  "route_id": 208,
  "direction": 0,
  "schedule": {
    "up": ["06:00", "06:15", "06:30", ...],
    "down": ["06:05", "06:20", "06:35", ...]
  },
  "statistics": {
    "up_dispatch_count": 73,
    "down_dispatch_count": 73,
    "total_dispatch": 146
  }
}
```

#### 评估结果文件 (JSON)
```json
{
  "dispatch_count_up": 73,
  "dispatch_count_down": 73,
  "total_dispatch": 146,
  "avg_waiting_time": 3.7,
  "stranded_passengers": 0,
  "total_passengers_served": 3157
}
```

---

## 常见问题

### Q1: CUDA out of memory 错误

**解决方案：**
1. 减小批次大小：编辑配置文件，将 `batch_size` 从64改为32或16
2. 使用CPU训练：`--device cpu`
3. 减少隐藏层神经元数量（不推荐，会影响性能）

### Q2: 训练速度很慢

**解决方案：**
1. 确认使用GPU：检查输出中是否显示 "使用设备: cuda"
2. 减少episode数进行快速测试：`--episodes 10`
3. 检查GPU利用率：`nvidia-smi`

### Q3: 模块导入错误

**解决方案：**
1. 确保在项目根目录运行脚本
2. 检查Python路径：`echo $PYTHONPATH`
3. 重新安装依赖：`pip install -r requirements.txt`

### Q4: 数据文件找不到

**解决方案：**
1. 确认 `test_data/` 目录存在
2. 检查数据文件路径是否正确
3. 运行测试：`python tests/test_modules.py`

### Q5: 训练结果与论文不一致

**可能原因：**
1. 随机种子不同：使用 `--seed 42` 确保一致性
2. 训练轮数不足：增加 `--episodes`
3. 超参数设置不同：检查配置文件

### Q6: 如何在没有GPU的机器上运行

**解决方案：**
```bash
# 使用CPU训练（会很慢）
python main.py train --route 208 --direction 0 --device cpu --episodes 10

# 或者只评估已训练的模型
python main.py evaluate --model results/models/route_208_dir_0.pth --route 208 --direction 0
```

---

## 性能优化建议

### 1. GPU加速
- 确保安装了正确版本的PyTorch（支持CUDA 11.8）
- 使用 `nvidia-smi` 监控GPU使用情况
- 调整批次大小以充分利用GPU内存

### 2. 训练加速
- 使用较大的批次大小（如果GPU内存允许）
- 减少学习频率（但可能影响性能）
- 使用混合精度训练（需要修改代码）

### 3. 内存优化
- 及时清理不需要的变量
- 使用较小的经验回放池
- 定期保存检查点并清理内存

---

## 联系和支持

如果遇到问题：
1. 查看本文档的常见问题部分
2. 运行测试脚本诊断问题
3. 检查错误日志

---

**最后更新**: 2025年1月
