"""时刻表生成器"""
import numpy as np
from typing import Dict, List, Tuple
from models.dqn_agent import DQNAgent
from environment.bus_env import BusEnvironment


class ScheduleGenerator:
    """时刻表生成器"""
    
    def __init__(self, agent: DQNAgent, env: BusEnvironment):
        """
        初始化时刻表生成器
        
        Args:
            agent: 训练好的DQN智能体
            env: 公交仿真环境
        """
        self.agent = agent
        self.env = env
    
    def generate_schedule(self) -> Dict[str, List[int]]:
        """
        生成公交时刻表
        
        Returns:
            {'up': [发车时间列表], 'down': [发车时间列表]}
        """
        state = self.env.reset()
        schedule_up = []
        schedule_down = []
        done = False
        
        print("开始生成时刻表...")
        
        while not done:
            # 使用贪婪策略选择动作（epsilon=0）
            action_idx = self.agent.select_action(state, epsilon=0.0)
            action = self.agent.action_index_to_tuple(action_idx)
            
            # 记录发车时间
            if action[0] == 1:
                schedule_up.append(self.env.current_time)
            if action[1] == 1:
                schedule_down.append(self.env.current_time)
            
            # 执行动作
            next_state, reward, done, info = self.env.step(action)
            state = next_state
        
        print(f"时刻表生成完成！")
        print(f"  上行发车次数: {len(schedule_up)}")
        print(f"  下行发车次数: {len(schedule_down)}")
        
        # 后处理：平衡上下行发车次数
        if len(schedule_up) != len(schedule_down):
            print(f"发车次数不平衡，进行调整...")
            schedule_up, schedule_down = self._balance_schedules(
                schedule_up, schedule_down
            )
            print(f"  调整后上行发车次数: {len(schedule_up)}")
            print(f"  调整后下行发车次数: {len(schedule_down)}")
        
        return {'up': schedule_up, 'down': schedule_down}
    
    def _balance_schedules(self, schedule_up: List[int],
                          schedule_down: List[int]) -> Tuple[List[int], List[int]]:
        """
        平衡上下行发车次数
        
        算法：
        1. 找到发车次数较多的方向
        2. 删除该方向的倒数第二次发车
        3. 从后向前调整发车时间，确保间隔不超过T_max
        
        Args:
            schedule_up: 上行时刻表
            schedule_down: 下行时刻表
            
        Returns:
            平衡后的(schedule_up, schedule_down)
        """
        if len(schedule_up) == len(schedule_down):
            return schedule_up, schedule_down
        
        # 复制列表避免修改原始数据
        schedule_up = schedule_up.copy()
        schedule_down = schedule_down.copy()
        
        # 确定需要调整的方向
        if len(schedule_up) > len(schedule_down):
            schedule = schedule_up
            is_up = True
        else:
            schedule = schedule_down
            is_up = False
        
        # 删除倒数第二次发车
        if len(schedule) >= 2:
            schedule.pop(-2)
        
        # 向前调整发车时间
        k = len(schedule) - 1
        while k > 0:
            if schedule[k] - schedule[k-1] > self.env.t_max:
                # 将前一次发车时间推迟
                schedule[k-1] = schedule[k] - self.env.t_max
            k -= 1
        
        # 返回调整后的时刻表
        if is_up:
            return schedule, schedule_down
        else:
            return schedule_up, schedule
    
    def format_schedule(self, schedule: Dict[str, List[int]]) -> Dict[str, List[str]]:
        """
        格式化时刻表为可读格式
        
        Args:
            schedule: 时刻表字典
            
        Returns:
            格式化后的时刻表 {'up': ['HH:MM', ...], 'down': ['HH:MM', ...]}
        """
        formatted = {}
        
        for direction, times in schedule.items():
            formatted[direction] = []
            for time_minute in times:
                hour = time_minute // 60
                minute = time_minute % 60
                formatted[direction].append(f"{hour:02d}:{minute:02d}")
        
        return formatted
    
    def save_schedule(self, schedule: Dict[str, List[int]], filepath: str):
        """
        保存时刻表到文件
        
        Args:
            schedule: 时刻表字典
            filepath: 保存路径
        """
        import json
        
        # 格式化时刻表
        formatted = self.format_schedule(schedule)
        
        # 添加统计信息
        output = {
            'route_id': self.env.route_id,
            'direction': self.env.direction,
            'schedule': formatted,
            'statistics': {
                'up_dispatch_count': len(schedule['up']),
                'down_dispatch_count': len(schedule['down']),
                'total_dispatch': len(schedule['up']) + len(schedule['down'])
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"时刻表已保存到: {filepath}")


if __name__ == '__main__':
    print("时刻表生成器模块")
    print("此模块需要配合训练好的模型和环境使用")
