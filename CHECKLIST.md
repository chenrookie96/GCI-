# DRL-TSBC 项目检查清单

## 使用前检查

### 环境检查
- [ ] Python 3.8+ 已安装
- [ ] CUDA 11.8 已安装（GPU训练）
- [ ] 虚拟环境已创建
- [ ] 依赖包已安装 (`pip install -r requirements.txt`)
- [ ] GPU可用性已验证 (`python -c "import torch; print(torch.cuda.is_available())"`)

### 数据检查
- [ ] test_data/208/ 目录存在
- [ ] test_data/208/passenger_dataframe_direction0.csv 存在
- [ ] test_data/208/passenger_dataframe_direction1.csv 存在
- [ ] test_data/208/traffic-0.csv 存在
- [ ] test_data/208/traffic-1.csv 存在
- [ ] test_data/211/ 目录存在（如果要训练211线路）

### 目录结构检查
- [ ] data/ 目录存在
- [ ] environment/ 目录存在
- [ ] models/ 目录存在
- [ ] inference/ 目录存在
- [ ] visualization/ 目录存在
- [ ] utils/ 目录存在
- [ ] experiments/ 目录存在
- [ ] results/ 目录存在
- [ ] configs/ 目录存在

### 配置文件检查
- [ ] configs/route_208_dir_0.json 存在
- [ ] configs/route_208_dir_1.json 存在
- [ ] configs/route_211_dir_0.json 存在
- [ ] configs/route_211_dir_1.json 存在

### 模块测试
- [ ] 运行 `python tests/test_modules.py` 全部通过

---

## 训练前检查

### 硬件资源
- [ ] GPU显存 >= 2GB（推荐4GB+）
- [ ] 系统内存 >= 8GB
- [ ] 磁盘空间 >= 5GB

### 训练参数
- [ ] 确认线路编号（208或211）
- [ ] 确认方向（0=上行，1=下行）
- [ ] 确认训练轮数（默认50）
- [ ] 确认设备（cuda或cpu）
- [ ] 确认随机种子（默认42）

### 预期时间
- [ ] GPU训练：30-60分钟/50 episodes
- [ ] CPU训练：2-4小时/50 episodes

---

## 训练后检查

### 输出文件
- [ ] results/models/route_X_dir_Y.pth 已生成
- [ ] results/models/route_X_dir_Y_training_data.json 已生成
- [ ] results/figures/convergence_route_X_dir_Y.png 已生成
- [ ] results/figures/loss_route_X_dir_Y.png 已生成

### 训练质量
- [ ] 训练收敛曲线平滑
- [ ] 上下行发车次数接近
- [ ] 平均奖励趋于稳定
- [ ] 损失值下降

---

## 评估前检查

### 模型文件
- [ ] 模型文件存在
- [ ] 模型文件大小正常（约200-300MB）
- [ ] 配置参数正确

---

## 评估后检查

### 输出文件
- [ ] results/models/route_X_dir_Y_schedule.json 已生成
- [ ] results/models/route_X_dir_Y_evaluation.json 已生成
- [ ] results/figures/schedule_heatmap_route_X_dir_Y.png 已生成

### 评估结果
- [ ] 发车次数合理（60-80次）
- [ ] 平均等待时间合理（3-5分钟）
- [ ] 滞留乘客数为0或很少
- [ ] 上下行发车次数相等

---

## 结果验证

### 与论文对比（208线路）
- [ ] 上行发车次数 ≈ 73
- [ ] 下行发车次数 ≈ 73
- [ ] 上行平均等待时间 ≈ 3.7分钟
- [ ] 下行平均等待时间 ≈ 3.8分钟
- [ ] 滞留乘客数 = 0

### 与论文对比（211线路）
- [ ] 上行发车次数 ≈ 75
- [ ] 下行发车次数 ≈ 75
- [ ] 上行平均等待时间 ≈ 4.0分钟
- [ ] 下行平均等待时间 ≈ 3.3分钟
- [ ] 滞留乘客数 = 0

---

## 可视化检查

### 图表生成
- [ ] 训练收敛曲线清晰
- [ ] 损失曲线下降趋势明显
- [ ] 时刻表热力图合理
- [ ] 中文标签显示正常

### 表格生成
- [ ] results/tables/comparison_route_208.csv 存在
- [ ] results/tables/comparison_route_211.csv 存在
- [ ] results/tables/omega_sensitivity.csv 存在
- [ ] 表格数据完整

---

## 常见问题排查

### 如果训练失败
- [ ] 检查CUDA是否可用
- [ ] 检查数据文件是否存在
- [ ] 检查磁盘空间是否充足
- [ ] 检查内存是否充足
- [ ] 查看错误日志

### 如果结果不理想
- [ ] 增加训练轮数
- [ ] 调整学习率
- [ ] 修改随机种子
- [ ] 检查数据质量
- [ ] 调整超参数

### 如果GPU内存不足
- [ ] 减小批次大小（64 -> 32 -> 16）
- [ ] 使用CPU训练
- [ ] 关闭其他GPU程序
- [ ] 减少经验池大小

---

## 提交前检查（如果需要提交作业）

### 代码完整性
- [ ] 所有Python文件存在
- [ ] 所有配置文件存在
- [ ] README.md 完整
- [ ] 代码注释充分

### 结果完整性
- [ ] 所有模型文件已保存
- [ ] 所有图表已生成
- [ ] 所有表格已生成
- [ ] 训练日志已保存

### 文档完整性
- [ ] README.md 说明清晰
- [ ] USAGE_GUIDE.md 详细
- [ ] PROJECT_SUMMARY.md 完整
- [ ] 代码注释充分

### 可复现性
- [ ] 随机种子已固定
- [ ] 配置文件已保存
- [ ] 依赖版本已记录
- [ ] 运行步骤已文档化

---

## 最终检查

- [ ] 运行 `python tests/test_modules.py` 全部通过
- [ ] 运行 `python quick_start.py` 成功完成
- [ ] 所有必需文件已生成
- [ ] 结果与论文基本一致
- [ ] 文档完整清晰
- [ ] 代码可以在其他机器上运行

---

## 备注

- 每完成一项，在 [ ] 中打勾 [x]
- 如果某项不适用，标记为 [N/A]
- 遇到问题，查看 USAGE_GUIDE.md 的常见问题部分
- 保持检查清单更新

---

**最后更新**: 2025年1月
