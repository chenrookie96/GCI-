# DRL-TSBC 实验脚本使用指南

本目录包含DRL-TSBC算法的训练和评估脚本。

## 文件说明

- `train_drl_tsbc.py`: 训练脚本
- `evaluate_drl_tsbc.py`: 评估脚本

## 使用方法

### 1. 训练模型

基本训练（使用默认参数）：
```bash
python experiments/train_drl_tsbc.py
```

自定义参数训练：
```bash
python experiments/train_drl_tsbc.py --route 208 --episodes 500 --save-interval 50
```

参数说明：
- `--route`: 路线编号 (208/211/683)
- `--episodes`: 训练轮数 (默认500，每个episode约15小时运营时间)
- `--save-interval`: 模型保存间隔 (默认50)
- `--device`: 训练设备 (auto/cuda/cpu)

### 2. 评估模型

```bash
python experiments/evaluate_drl_tsbc.py --model results/208_xxx/model_final.pth --route 208
```

参数说明：
- `--model`: 模型文件路径
- `--route`: 路线编号

## 训练输出

训练完成后，结果保存在 `results/` 目录下：

```
results/
└── 208_20251017_123456/
    ├── model_final.pth          # 最终模型
    ├── model_episode_100.pth    # 检查点
    ├── training_curves.png      # 训练曲线
    └── training_metrics.json    # 训练指标
```

## 评估输出

评估脚本会输出：
- 时刻表生成结果
- 性能指标对比
- 平衡调整效果

## 论文规范参数

所有实现严格按照论文表2-2规范：
- 学习率: 0.001
- 折现系数γ: 0.4
- Epsilon: 0.1 (固定)
- 批次大小: 64
- 经验池大小: 3000
- 学习频率: 5
- 目标网络更新频率: 100
- Tmin: 3分钟
- Tmax: 15分钟

## 注意事项

1. 默认训练500个episodes，每个episode是一整天的运营周期（约900分钟）
2. 首次训练建议使用较少的episodes（如50-100）测试环境配置
3. GPU训练速度显著快于CPU
4. 训练过程中会自动保存检查点，可以随时中断和恢复
5. 评估时使用贪婪策略（epsilon=0）
6. 根据论文，模型通常在200-500个episodes后收敛
