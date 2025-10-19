"""评估模块"""
import numpy as np
import pandas as pd
from typing import Dict, List
from environment.bus_env import BusEnvironment


class ScheduleEvaluator:
    """时刻表评估器"""
    
    def __init__(self, env: BusEnvironment):
        """
        初始化评估器
        
        Args:
            env: 公交仿真环境
        """
        self.env = env
    
    def evaluate_schedule(self, schedule: Dict[str, List[int]]) -> Dict:
        """
        评估时刻表性能
        
        Args:
            schedule: 时刻表字典 {'up': [...], 'down': [...]}
            
        Returns:
            性能指标字典
        """
        print("开始评估时刻表...")
        
        # 重置环境
        state = self.env.reset()
        
        # 统计变量
        total_waiting_time_up = 0
        total_waiting_time_down = 0
        total_passengers_up = 0
        total_passengers_down = 0
        stranded_passengers_up = 0
        stranded_passengers_down = 0
        
        # 创建发车时间集合以便快速查找
        schedule_up_set = set(schedule['up'])
        schedule_down_set = set(schedule['down'])
        
        done = False
        step_count = 0
        
        while not done:
            # 根据时刻表决定是否发车
            current_time = self.env.current_time
            action = (
                1 if current_time in schedule_up_set else 0,
                1 if current_time in schedule_down_set else 0
            )
            
            # 执行动作
            next_state, reward, done, info = self.env.step(action)
            
            # 收集统计数据（简化版本）
            step_count += 1
            
            state = next_state
        
        # 计算最终指标
        # 注意：这里使用简化的计算方式，实际应该在环境中收集详细数据
        
        # 从环境获取统计信息
        total_passengers = self.env.total_passengers_served
        avg_waiting_time = self.env.total_waiting_time / max(total_passengers, 1)
        stranded_count = len(self.env.stranded_passengers)
        
        results = {
            'dispatch_count_up': len(schedule['up']),
            'dispatch_count_down': len(schedule['down']),
            'total_dispatch': len(schedule['up']) + len(schedule['down']),
            'avg_waiting_time': avg_waiting_time / 60.0,  # 转换为分钟
            'stranded_passengers': stranded_count,
            'total_passengers_served': total_passengers,
            'steps': step_count
        }
        
        print(f"评估完成！")
        print(f"  上行发车次数: {results['dispatch_count_up']}")
        print(f"  下行发车次数: {results['dispatch_count_down']}")
        print(f"  平均等待时间: {results['avg_waiting_time']:.2f} 分钟")
        print(f"  滞留乘客数: {results['stranded_passengers']}")
        print(f"  服务乘客数: {results['total_passengers_served']}")
        
        return results
    
    def compute_capacity_over_time(self, schedule: Dict[str, List[int]],
                                   time_interval: int = 30) -> pd.DataFrame:
        """
        计算每个时间段的客运容量
        
        Args:
            schedule: 时刻表字典
            time_interval: 时间间隔（分钟）
            
        Returns:
            包含时间段和客运容量的DataFrame
        """
        print(f"计算客运容量（时间间隔: {time_interval}分钟）...")
        
        # 重置环境
        state = self.env.reset()
        
        # 初始化数据收集
        time_slots = []
        capacity_up = []
        capacity_down = []
        demand_up = []
        demand_down = []
        
        current_slot_start = self.env.start_time
        current_slot_capacity_up = 0
        current_slot_capacity_down = 0
        current_slot_demand_up = 0
        current_slot_demand_down = 0
        
        # 创建发车时间集合
        schedule_up_set = set(schedule['up'])
        schedule_down_set = set(schedule['down'])
        
        done = False
        
        while not done:
            current_time = self.env.current_time
            
            # 检查是否进入新的时间段
            if current_time >= current_slot_start + time_interval:
                # 保存当前时间段的数据
                time_slots.append(current_slot_start)
                capacity_up.append(current_slot_capacity_up)
                capacity_down.append(current_slot_capacity_down)
                demand_up.append(current_slot_demand_up)
                demand_down.append(current_slot_demand_down)
                
                # 重置计数器
                current_slot_start = current_time
                current_slot_capacity_up = 0
                current_slot_capacity_down = 0
                current_slot_demand_up = 0
                current_slot_demand_down = 0
            
            # 根据时刻表发车
            action = (
                1 if current_time in schedule_up_set else 0,
                1 if current_time in schedule_down_set else 0
            )
            
            # 统计该时刻的客运容量（简化：每辆车提供固定容量）
            if action[0] == 1:
                current_slot_capacity_up += self.env.max_capacity
            if action[1] == 1:
                current_slot_capacity_down += self.env.max_capacity
            
            # 统计需求（简化：使用等待乘客数）
            for station, passengers in self.env.waiting_passengers.items():
                current_slot_demand_up += len(passengers)
            
            # 执行动作
            next_state, reward, done, info = self.env.step(action)
            state = next_state
        
        # 保存最后一个时间段
        if current_slot_start < self.env.end_time:
            time_slots.append(current_slot_start)
            capacity_up.append(current_slot_capacity_up)
            capacity_down.append(current_slot_capacity_down)
            demand_up.append(current_slot_demand_up)
            demand_down.append(current_slot_demand_down)
        
        # 创建DataFrame
        df = pd.DataFrame({
            'time_minute': time_slots,
            'time_str': [f"{t//60:02d}:{t%60:02d}" for t in time_slots],
            'capacity_up': capacity_up,
            'capacity_down': capacity_down,
            'demand_up': demand_up,
            'demand_down': demand_down,
            'total_capacity': [u + d for u, d in zip(capacity_up, capacity_down)],
            'total_demand': [u + d for u, d in zip(demand_up, demand_down)]
        })
        
        print(f"客运容量计算完成！共 {len(df)} 个时间段")
        
        return df
    
    def compare_schedules(self, schedules: Dict[str, Dict[str, List[int]]]) -> pd.DataFrame:
        """
        对比多个时刻表的性能
        
        Args:
            schedules: {'方法名': 时刻表字典}
            
        Returns:
            对比结果DataFrame
        """
        print("对比多个时刻表...")
        
        results = []
        
        for method_name, schedule in schedules.items():
            print(f"\n评估 {method_name}...")
            metrics = self.evaluate_schedule(schedule)
            metrics['method'] = method_name
            results.append(metrics)
        
        df = pd.DataFrame(results)
        
        print("\n对比完成！")
        return df


if __name__ == '__main__':
    print("评估模块")
    print("此模块需要配合环境和时刻表使用")
