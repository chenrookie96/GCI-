# DRL-TSBC 算法复现项目

基于深度强化学习的双向动态公交时刻表排班算法（DRL-TSBC）复现实现。

## 项目简介

本项目复现了谢嘉昊论文中提出的DRL-TSBC算法，该算法使用深度Q网络（DQN）解决公交时刻表排班问题，通过强化学习动态决策发车时间，并保证上下行方向发车次数平衡。

## 环境要求

- Python 3.8+
- CUDA 11.8 (用于GPU加速)
- GPU: NVIDIA RTX 4060 或更高

## 安装步骤

1. 克隆项目
```bash
git clone <repository-url>
cd drl-tsbc
```

2. 创建虚拟环境（推荐）
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

4. 验证CUDA可用性
```python
import torch
print(torch.cuda.is_available())  # 应该输出 True
print(torch.cuda.get_device_name(0))  # 显示GPU型号
```

## 项目结构

```
drl-tsbc/
├── data/                   # 数据加载模块
├── environment/            # 公交仿真环境
├── models/                 # DQN网络和训练
├── inference/              # 推理和评估
├── visualization/          # 可视化
├── utils/                  # 工具和配置
├── experiments/            # 实验脚本
├── tests/                  # 测试
├── results/                # 实验结果
│   ├── models/            # 保存的模型
│   ├── logs/              # 训练日志
│   ├── figures/           # 生成的图表
│   └── tables/            # 生成的表格
├── configs/                # 配置文件
└── test_data/              # 测试数据集
```

## 快速开始

### 方式1: 使用快速开始脚本（推荐）

最简单的方式，自动训练和评估208线路：

```bash
python quick_start.py
```

### 方式2: 使用主程序

训练单个模型：
```bash
# 训练208线路上行
python main.py train --route 208 --direction 0 --episodes 50

# 训练211线路下行
python main.py train --route 211 --direction 1 --episodes 50
```

评估模型：
```bash
python main.py evaluate --model results/models/route_208_dir_0.pth --route 208 --direction 0
```

### 方式3: 批量训练所有线路

```bash
python experiments/train_all.py --episodes 50
```

### 方式4: 使用配置文件

```bash
# 使用预定义的配置文件
python main.py train --config configs/route_208_dir_0.json
```

### 生成对比表格

```bash
python experiments/compare_methods.py
```

### 测试模块

```bash
python tests/test_modules.py
```

## 数据集

项目使用的数据集位于 `test_data/` 目录：
- 208线路：上行26站，下行24站
- 211线路：上行17站，下行11站

数据包括：
- 乘客数据：到达时间、上下车站点
- 交通数据：站间行驶时间

## 主要参数

### DQN网络参数
- 学习率: 0.001
- 隐藏层数: 12
- 每层神经元数: 500
- 批次大小: 64
- 折扣因子γ: 0.4
- 经验池大小: 3000

### 奖励函数参数
- μ (等待时间归一化): 5000
- δ (发车次数归一化): 200
- β (滞留乘客惩罚): 0.2
- ζ (发车平衡权重): 0.002
- ω (等待时间惩罚): 1/1000 (208线), 1/900 (211线)

## 实验结果

训练完成后，可以在 `results/` 目录查看：
- 训练收敛曲线
- 性能对比表格
- 时刻表可视化
- 参数敏感性分析

## 参考文献

谢嘉昊, 王玺钧. 基于交通大数据的公交排班和调度机制研究.

## 许可证

MIT License
