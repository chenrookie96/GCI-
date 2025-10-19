# DRL-TSBC 项目文件索引

## 快速导航

- [核心代码](#核心代码)
- [文档文件](#文档文件)
- [配置文件](#配置文件)
- [数据文件](#数据文件)
- [输出目录](#输出目录)

---

## 核心代码

### 数据处理模块 (data/)
| 文件 | 说明 | 行数 |
|------|------|------|
| `data/__init__.py` | 模块初始化 | 1 |
| `data/data_loader.py` | 乘客和交通数据加载器 | ~250 |

**主要类**:
- `PassengerDataLoader`: 加载和管理乘客数据
- `TrafficDataLoader`: 加载和管理交通数据

---

### 环境模块 (environment/)
| 文件 | 说明 | 行数 |
|------|------|------|
| `environment/__init__.py` | 模块初始化 | 1 |
| `environment/entities.py` | 实体类定义 | ~100 |
| `environment/bus_env.py` | 公交仿真环境 | ~400 |

**主要类**:
- `Passenger`: 乘客实体
- `Bus`: 公交车实体
- `BusEnvironment`: 完整的仿真环境

---

### DQN模块 (models/)
| 文件 | 说明 | 行数 |
|------|------|------|
| `models/__init__.py` | 模块初始化 | 1 |
| `models/dqn_network.py` | DQN网络架构 | ~100 |
| `models/replay_buffer.py` | 经验回放池 | ~80 |
| `models/dqn_agent.py` | DQN智能体 | ~300 |

**主要类**:
- `DQN`: 深度Q网络
- `ReplayBuffer`: 经验回放池
- `DQNAgent`: 完整的训练智能体

---

### 推理模块 (inference/)
| 文件 | 说明 | 行数 |
|------|------|------|
| `inference/__init__.py` | 模块初始化 | 1 |
| `inference/schedule_generator.py` | 时刻表生成器 | ~150 |
| `inference/evaluator.py` | 性能评估器 | ~200 |

**主要类**:
- `ScheduleGenerator`: 生成和后处理时刻表
- `ScheduleEvaluator`: 评估时刻表性能

---

### 可视化模块 (visualization/)
| 文件 | 说明 | 行数 |
|------|------|------|
| `visualization/__init__.py` | 模块初始化 | 1 |
| `visualization/visualizer.py` | 可视化工具 | ~350 |

**主要类**:
- `Visualizer`: 生成各种图表和表格

**支持的图表**:
- 训练收敛曲线
- 损失曲线
- 参数敏感性分析图
- 客运容量对比图
- 时刻表热力图

---

### 工具模块 (utils/)
| 文件 | 说明 | 行数 |
|------|------|------|
| `utils/__init__.py` | 模块初始化 | 1 |
| `utils/config.py` | 配置管理 | ~200 |
| `utils/helpers.py` | 辅助工具函数 | ~200 |

**主要类**:
- `RouteConfig`: 线路配置
- `DQNConfig`: DQN配置
- `ExperimentConfig`: 实验配置
- `ConfigManager`: 配置管理器

**辅助函数**:
- `set_seed()`: 设置随机种子
- `check_cuda()`: 检查CUDA
- `format_time()`: 时间格式化
- `get_device()`: 获取设备

---

### 实验脚本 (experiments/)
| 文件 | 说明 | 行数 |
|------|------|------|
| `experiments/__init__.py` | 模块初始化 | 1 |
| `experiments/train.py` | 训练脚本 | ~200 |
| `experiments/evaluate.py` | 评估脚本 | ~150 |
| `experiments/train_all.py` | 批量训练脚本 | ~150 |
| `experiments/compare_methods.py` | 对比实验脚本 | ~200 |

**用途**:
- `train.py`: 训练单个模型
- `evaluate.py`: 评估模型性能
- `train_all.py`: 批量训练所有线路
- `compare_methods.py`: 生成对比表格

---

### 测试模块 (tests/)
| 文件 | 说明 | 行数 |
|------|------|------|
| `tests/test_modules.py` | 模块测试 | ~250 |

**测试内容**:
- 模块导入测试
- 配置正确性测试
- 数据文件测试
- 目录结构测试
- DQN网络测试

---

### 主程序
| 文件 | 说明 | 行数 |
|------|------|------|
| `main.py` | 主入口程序 | ~100 |
| `quick_start.py` | 快速开始脚本 | ~150 |

**用途**:
- `main.py`: 统一的命令行入口
- `quick_start.py`: 一键训练和评估

---

## 文档文件

### 项目文档
| 文件 | 说明 | 页数 |
|------|------|------|
| `README.md` | 项目说明 | 3 |
| `USAGE_GUIDE.md` | 详细使用指南 | 15 |
| `PROJECT_SUMMARY.md` | 项目总结 | 10 |
| `PROJECT_COMPLETION_REPORT.md` | 完成报告 | 12 |
| `CHECKLIST.md` | 检查清单 | 8 |
| `FILE_INDEX.md` | 文件索引（本文件） | 5 |

### 算法文档
| 文件 | 说明 | 来源 |
|------|------|------|
| `参数设置.md` | 参数配置说明 | 论文整理 |
| `实验结果和可视化要求.md` | 实验要求 | 论文整理 |
| `文章算法和公式.md` | 算法详解 | 论文整理 |
| `算法流程.md` | 算法流程 | 论文整理 |
| `request.md` | 项目需求 | 用户提供 |

### Spec文档
| 文件 | 说明 | 位置 |
|------|------|------|
| `requirements.md` | 需求文档 | `.kiro/specs/drl-tsbc-algorithm/` |
| `design.md` | 设计文档 | `.kiro/specs/drl-tsbc-algorithm/` |
| `tasks.md` | 任务列表 | `.kiro/specs/drl-tsbc-algorithm/` |

---

## 配置文件

### 线路配置
| 文件 | 说明 |
|------|------|
| `configs/route_208_dir_0.json` | 208线路上行配置 |
| `configs/route_208_dir_1.json` | 208线路下行配置 |
| `configs/route_211_dir_0.json` | 211线路上行配置 |
| `configs/route_211_dir_1.json` | 211线路下行配置 |

### 其他配置
| 文件 | 说明 |
|------|------|
| `requirements.txt` | Python依赖包 |
| `.gitignore` | Git忽略配置 |

---

## 数据文件

### 208线路数据
```
test_data/208/
├── passenger_dataframe_direction0.csv    # 上行乘客数据
├── passenger_dataframe_direction1.csv    # 下行乘客数据
├── traffic-0.csv                         # 上行交通数据
└── traffic-1.csv                         # 下行交通数据
```

### 211线路数据
```
test_data/211/
├── passenger_dataframe_direction0.csv    # 上行乘客数据
├── passenger_dataframe_direction1.csv    # 下行乘客数据
├── traffic-0.csv                         # 上行交通数据
└── traffic-1.csv                         # 下行交通数据
```

### 数据说明
| 文件 | 说明 |
|------|------|
| `test_data/数据说明.txt` | 数据格式说明 |

---

## 输出目录

### 模型文件 (results/models/)
```
results/models/
├── route_208_dir_0.pth                    # 模型权重
├── route_208_dir_0_training_data.json     # 训练数据
├── route_208_dir_0_schedule.json          # 时刻表
├── route_208_dir_0_evaluation.json        # 评估结果
└── ... (其他线路类似)
```

### 图表文件 (results/figures/)
```
results/figures/
├── convergence_route_208_dir_0.png        # 收敛曲线
├── loss_route_208_dir_0.png               # 损失曲线
├── schedule_heatmap_route_208_dir_0.png   # 时刻表热力图
└── ... (其他线路类似)
```

### 表格文件 (results/tables/)
```
results/tables/
├── comparison_route_208.csv               # 208线路对比
├── comparison_route_211.csv               # 211线路对比
├── comparison_all_routes.csv              # 全部对比
├── omega_sensitivity.csv                  # 参数敏感性
└── training_summary.csv                   # 训练总结
```

### 日志文件 (results/logs/)
```
results/logs/
└── (训练日志，如果启用)
```

---

## 文件关系图

```
main.py / quick_start.py
    │
    ├─→ experiments/train.py
    │       │
    │       ├─→ utils/config.py
    │       ├─→ data/data_loader.py
    │       ├─→ environment/bus_env.py
    │       ├─→ models/dqn_agent.py
    │       └─→ visualization/visualizer.py
    │
    └─→ experiments/evaluate.py
            │
            ├─→ inference/schedule_generator.py
            ├─→ inference/evaluator.py
            └─→ visualization/visualizer.py
```

---

## 使用频率

### 高频使用
- `main.py` - 主入口
- `quick_start.py` - 快速开始
- `README.md` - 项目说明
- `USAGE_GUIDE.md` - 使用指南

### 中频使用
- `experiments/train.py` - 训练
- `experiments/evaluate.py` - 评估
- `utils/config.py` - 配置
- `tests/test_modules.py` - 测试

### 低频使用
- `experiments/train_all.py` - 批量训练
- `experiments/compare_methods.py` - 对比实验
- `PROJECT_SUMMARY.md` - 项目总结
- `CHECKLIST.md` - 检查清单

---

## 文件大小估算

### 代码文件
- Python文件: ~3500行，约150KB
- 配置文件: ~2KB

### 文档文件
- Markdown文件: ~100页，约500KB

### 数据文件
- 乘客数据: ~5MB
- 交通数据: ~100KB

### 输出文件
- 模型文件: ~200-300MB/模型
- 图表文件: ~1-2MB/图
- 表格文件: ~10-50KB/表

---

## 快速查找

### 我想...

**训练模型**
→ 查看 `main.py` 或 `experiments/train.py`

**评估模型**
→ 查看 `experiments/evaluate.py`

**修改参数**
→ 查看 `utils/config.py` 或 `configs/*.json`

**理解算法**
→ 查看 `文章算法和公式.md` 或 `environment/bus_env.py`

**生成图表**
→ 查看 `visualization/visualizer.py`

**测试代码**
→ 查看 `tests/test_modules.py`

**解决问题**
→ 查看 `USAGE_GUIDE.md` 的常见问题部分

**了解项目**
→ 查看 `README.md` 和 `PROJECT_SUMMARY.md`

---

**最后更新**: 2025年1月  
**文件总数**: 50+  
**代码行数**: 3500+  
**文档页数**: 100+
