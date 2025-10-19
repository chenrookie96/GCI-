"""可视化模块"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from typing import List, Dict


class Visualizer:
    """可视化工具类"""
    
    def __init__(self, output_dir: str = 'results/figures'):
        """
        初始化可视化工具
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 设置样式
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['figure.dpi'] = 100
    
    def plot_training_convergence(self, episode_rewards: List[float],
                                  dispatch_counts: List[Dict],
                                  title: str = 'DRL-TSBC 收敛结果',
                                  filename: str = 'convergence.png'):
        """
        绘制训练收敛曲线（图2-11）
        
        Args:
            episode_rewards: 每个episode的平均奖励
            dispatch_counts: 每个episode的发车次数 [{'up': x, 'down': y}, ...]
            title: 图表标题
            filename: 保存文件名
        """
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        episodes = range(1, len(episode_rewards) + 1)
        
        # 左Y轴：平均奖励
        color = 'tab:blue'
        ax1.set_xlabel('Episode', fontsize=12)
        ax1.set_ylabel('平均奖励', color=color, fontsize=12)
        ax1.plot(episodes, episode_rewards, color=color, linewidth=2, label='平均奖励')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, alpha=0.3)
        
        # 右Y轴：发车次数
        ax2 = ax1.twinx()
        color_up = 'tab:orange'
        color_down = 'tab:purple'
        ax2.set_ylabel('发车次数', fontsize=12)
        
        dispatch_up = [d['up'] for d in dispatch_counts]
        dispatch_down = [d['down'] for d in dispatch_counts]
        
        ax2.plot(episodes, dispatch_up, color=color_up, linewidth=2, 
                marker='o', markersize=4, label='上行发车次数')
        ax2.plot(episodes, dispatch_down, color=color_down, linewidth=2,
                marker='s', markersize=4, label='下行发车次数')
        
        # 图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=10)
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"训练收敛曲线已保存到: {filepath}")
    
    def plot_omega_sensitivity(self, omega_values: List[float],
                              results: List[Dict],
                              route_id: int,
                              filename: str = 'omega_sensitivity.png'):
        """
        绘制ω参数敏感性分析图（图2-8）
        
        Args:
            omega_values: ω值列表
            results: 结果列表，每个元素包含性能指标
            route_id: 线路编号
            filename: 保存文件名
        """
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # 提取数据
        dispatch_counts = [r['total_dispatch'] for r in results]
        awt_up = [r.get('avg_waiting_time_up', 0) for r in results]
        awt_down = [r.get('avg_waiting_time_down', 0) for r in results]
        
        # 格式化ω值标签
        omega_labels = [f'1/{int(1/w)}' for w in omega_values]
        
        # 左Y轴：发车次数
        color = 'tab:blue'
        ax1.set_xlabel('ω', fontsize=12)
        ax1.set_ylabel('发车次数 (NDT)', color=color, fontsize=12)
        ax1.plot(omega_labels, dispatch_counts, color=color, linewidth=2,
                marker='o', markersize=8, label='发车次数')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, alpha=0.3)
        
        # 右Y轴：平均等待时间
        ax2 = ax1.twinx()
        color_up = 'tab:orange'
        color_down = 'tab:purple'
        ax2.set_ylabel('乘客平均等待时间 (分钟)', fontsize=12)
        
        ax2.plot(omega_labels, awt_up, color=color_up, linewidth=2,
                marker='s', markersize=8, label='上行AWT')
        ax2.plot(omega_labels, awt_down, color=color_down, linewidth=2,
                marker='^', markersize=8, label='下行AWT')
        
        # 图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=10)
        
        plt.title(f'DRL-TSBC 在不同 ω 下的 {route_id} 线发车次数与乘客平均等待时间的对比',
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"参数敏感性分析图已保存到: {filepath}")
    
    def plot_capacity_comparison(self, time_points: List[int],
                                capacity_data: Dict[str, List[float]],
                                title: str,
                                filename: str,
                                ylabel: str = '总客运量'):
        """
        绘制客运容量对比图（图2-3, 2-4, 2-5, 2-6, 2-7）
        
        Args:
            time_points: 时间点列表（分钟）
            capacity_data: {'标签': 容量列表}
            title: 图表标题
            filename: 保存文件名
            ylabel: Y轴标签
        """
        plt.figure(figsize=(14, 6))
        
        # 转换时间点为小时:分钟格式
        time_labels = [f"{t//60}:{t%60:02d}" for t in time_points]
        
        # 绘制每条曲线
        colors = ['tab:blue', 'tab:cyan', 'tab:orange', 'tab:green',
                 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink']
        
        for idx, (label, capacity) in enumerate(capacity_data.items()):
            color = colors[idx % len(colors)]
            plt.plot(time_points, capacity, label=label, linewidth=2,
                    color=color, alpha=0.8)
        
        plt.xlabel('时间', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(fontsize=10, loc='best')
        plt.grid(True, alpha=0.3)
        
        # 设置X轴刻度（每2小时显示一次）
        tick_indices = range(0, len(time_points), 120)  # 每120分钟
        plt.xticks([time_points[i] for i in tick_indices if i < len(time_points)],
                  [time_labels[i] for i in tick_indices if i < len(time_labels)],
                  rotation=45)
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"客运容量对比图已保存到: {filepath}")
    
    def generate_comparison_table(self, results: List[Dict],
                                 filename: str = 'comparison_table.csv') -> pd.DataFrame:
        """
        生成性能对比表格（表2-3）
        
        Args:
            results: 结果列表
            filename: 保存文件名
            
        Returns:
            对比结果DataFrame
        """
        df = pd.DataFrame(results)
        
        filepath = os.path.join(self.output_dir.replace('figures', 'tables'), filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        
        print(f"对比表格已保存到: {filepath}")
        
        return df
    
    def plot_loss_curve(self, losses: List[float], filename: str = 'loss_curve.png'):
        """
        绘制损失曲线
        
        Args:
            losses: 损失值列表
            filename: 保存文件名
        """
        plt.figure(figsize=(12, 6))
        
        # 平滑损失曲线
        window_size = min(100, len(losses) // 10)
        if window_size > 1:
            smoothed_losses = pd.Series(losses).rolling(window=window_size).mean()
            plt.plot(smoothed_losses, label='平滑损失', linewidth=2, color='tab:blue')
        
        plt.plot(losses, label='原始损失', linewidth=0.5, alpha=0.3, color='tab:gray')
        
        plt.xlabel('训练步数', fontsize=12)
        plt.ylabel('损失值', fontsize=12)
        plt.title('训练损失曲线', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"损失曲线已保存到: {filepath}")
    
    def plot_schedule_heatmap(self, schedule: Dict[str, List[int]],
                             filename: str = 'schedule_heatmap.png'):
        """
        绘制时刻表热力图
        
        Args:
            schedule: 时刻表字典
            filename: 保存文件名
        """
        # 创建24小时的时间槽
        hours = 24
        slots_per_hour = 4  # 每小时4个15分钟时间槽
        
        # 初始化热力图数据
        heatmap_data = np.zeros((2, hours * slots_per_hour))
        
        # 填充上行数据
        for time_minute in schedule['up']:
            hour = time_minute // 60
            minute = time_minute % 60
            slot = hour * slots_per_hour + minute // 15
            if slot < heatmap_data.shape[1]:
                heatmap_data[0, slot] += 1
        
        # 填充下行数据
        for time_minute in schedule['down']:
            hour = time_minute // 60
            minute = time_minute % 60
            slot = hour * slots_per_hour + minute // 15
            if slot < heatmap_data.shape[1]:
                heatmap_data[1, slot] += 1
        
        # 绘制热力图
        plt.figure(figsize=(16, 4))
        sns.heatmap(heatmap_data, cmap='YlOrRd', cbar_kws={'label': '发车次数'},
                   yticklabels=['上行', '下行'], xticklabels=False)
        
        # 设置X轴标签（每小时）
        xticks = range(0, hours * slots_per_hour, slots_per_hour)
        xticklabels = [f'{h:02d}:00' for h in range(hours)]
        plt.xticks(xticks, xticklabels, rotation=0)
        
        plt.title('公交发车时刻表热力图', fontsize=14, fontweight='bold')
        plt.xlabel('时间', fontsize=12)
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"时刻表热力图已保存到: {filepath}")


if __name__ == '__main__':
    print("可视化模块")
    print("此模块提供各种图表绘制功能")
    
    # 测试基本功能
    vis = Visualizer()
    print(f"输出目录: {vis.output_dir}")
