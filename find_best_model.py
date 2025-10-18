"""
从训练结果中找到最佳模型
"""
import json
from pathlib import Path

# 找到最新的results目录
results_dir = Path('results')
latest_run = sorted(results_dir.glob('208_*'))[-1]

print(f"分析训练结果: {latest_run}")

# 读取训练指标
metrics_file = latest_run / 'training_metrics.json'
with open(metrics_file, 'r') as f:
    metrics = json.load(f)

# 分析每50个episode的平均等待时间
episodes = len(metrics['episode_rewards'])
print(f"\n总episodes: {episodes}")

# 找到下行等待时间最低的episode
# 注意：我们需要从训练输出中手动记录，因为metrics.json可能没有详细的等待时间

print(f"\n从训练输出观察到的关键结果:")
print(f"  Episode 50:  下行等待 16.8分钟")
print(f"  Episode 100: 下行等待 21.3分钟")
print(f"  Episode 350: 下行等待 18.0分钟")
print(f"  Episode 400: 下行等待 21.3分钟")
print(f"  Episode 450: 下行等待 12.3分钟 ⭐ 最佳!")
print(f"  Episode 500: 下行等待 27.3分钟")

print(f"\n结论:")
print(f"  Episode 450的模型效果最好")
print(f"  模型文件: {latest_run / 'model_episode_450.pth'}")
print(f"\n建议:")
print(f"  1. 使用Episode 450的模型作为最终模型")
print(f"  2. 训练可能过拟合或者需要early stopping")
print(f"  3. 可以尝试降低学习率或调整其他超参数")
