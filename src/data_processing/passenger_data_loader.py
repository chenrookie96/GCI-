# src/data_processing/passenger_data_loader.py
"""
乘客数据加载器
用于加载和预处理公交乘客刷卡数据
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PassengerDataLoader:
    """
    乘客数据加载器
    
    支持加载208、211、683路线的乘客刷卡数据
    数据格式：Label, Boarding time, Boarding station, Alighting station, Arrival time
    """
    
    def __init__(self, data_dir: str = "test_data"):
        """
        初始化数据加载器
        
        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = Path(data_dir)
        self.routes = ['208', '211', '683']
        self.passenger_data = {}
        self.traffic_data = {}
        
    def load_route_data(self, route: str) -> Dict[str, pd.DataFrame]:
        """
        加载指定路线的数据
        
        Args:
            route: 路线编号 ('208', '211', '683')
            
        Returns:
            包含上下行数据的字典
        """
        if route not in self.routes:
            raise ValueError(f"不支持的路线: {route}. 支持的路线: {self.routes}")
        
        route_dir = self.data_dir / route
        if not route_dir.exists():
            raise FileNotFoundError(f"路线数据目录不存在: {route_dir}")
        
        logger.info(f"加载路线 {route} 的数据...")
        
        # 加载乘客数据
        passenger_data = {}
        for direction in [0, 1]:
            file_path = route_dir / f"passenger_dataframe_direction{direction}.csv"
            if not file_path.exists():
                raise FileNotFoundError(f"乘客数据文件不存在: {file_path}")
            
            df = pd.read_csv(file_path)
            self._validate_passenger_data(df, route, direction)
            passenger_data[f'direction_{direction}'] = df
            logger.info(f"  方向{direction}: 加载 {len(df)} 条乘客记录")
        
        # 加载交通数据
        traffic_data = {}
        for direction in [0, 1]:
            file_path = route_dir / f"traffic-{direction}.csv"
            if file_path.exists():
                df = pd.read_csv(file_path)
                traffic_data[f'direction_{direction}'] = df
                logger.info(f"  方向{direction}: 加载交通数据 {len(df)} 条记录")
        
        self.passenger_data[route] = passenger_data
        self.traffic_data[route] = traffic_data
        
        return {
            'passenger': passenger_data,
            'traffic': traffic_data
        }
    
    def load_all_routes(self) -> Dict[str, Dict]:
        """
        加载所有路线的数据
        
        Returns:
            所有路线的数据字典
        """
        all_data = {}
        for route in self.routes:
            try:
                all_data[route] = self.load_route_data(route)
            except Exception as e:
                logger.error(f"加载路线 {route} 失败: {e}")
                continue
        
        return all_data
    
    def _validate_passenger_data(self, df: pd.DataFrame, route: str, direction: int):
        """
        验证乘客数据格式
        
        Args:
            df: 乘客数据DataFrame
            route: 路线编号
            direction: 方向 (0或1)
        """
        required_columns = ['Label', 'Boarding time', 'Boarding station', 
                          'Alighting station', 'Arrival time']
        
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"路线{route}方向{direction}缺少必需列: {missing_columns}")
        
        # 检查数据类型和范围
        if df['Boarding time'].min() < 0 or df['Boarding time'].max() > 1440:
            logger.warning(f"路线{route}方向{direction}: 上车时间超出范围 [0, 1440]")
        
        if df['Arrival time'].min() < 0 or df['Arrival time'].max() > 1440:
            logger.warning(f"路线{route}方向{direction}: 到达时间超出范围 [0, 1440]")
        
        if (df['Boarding station'] < 0).any():
            logger.warning(f"路线{route}方向{direction}: 存在负数站点编号")
        
        if (df['Alighting station'] < 0).any():
            logger.warning(f"路线{route}方向{direction}: 存在负数站点编号")
    
    def preprocess_passenger_data(self, route: str, direction: int) -> pd.DataFrame:
        """
        预处理乘客数据
        
        Args:
            route: 路线编号
            direction: 方向 (0或1)
            
        Returns:
            预处理后的DataFrame
        """
        if route not in self.passenger_data:
            raise ValueError(f"路线 {route} 的数据未加载")
        
        df = self.passenger_data[route][f'direction_{direction}'].copy()
        
        # 按到达时间排序
        df = df.sort_values('Arrival time').reset_index(drop=True)
        
        # 添加时间特征
        df['hour'] = df['Boarding time'] // 60
        df['minute'] = df['Boarding time'] % 60
        df['arrival_hour'] = df['Arrival time'] // 60
        df['arrival_minute'] = df['Arrival time'] % 60
        
        # 计算等待时间
        df['waiting_time'] = df['Boarding time'] - df['Arrival time']
        df['waiting_time'] = df['waiting_time'].clip(lower=0)
        
        # 添加时段标记
        df['time_period'] = pd.cut(df['hour'], 
                                   bins=[0, 7, 9, 17, 19, 24],
                                   labels=['早晨', '早高峰', '平峰', '晚高峰', '晚间'])
        
        logger.info(f"路线{route}方向{direction}预处理完成: {len(df)}条记录")
        
        return df
    
    def estimate_arrival_times(self, boarding_times: np.ndarray, 
                              time_diff: Optional[np.ndarray] = None) -> np.ndarray:
        """
        估算乘客到达时间
        
        根据论文2.4.1节，使用正态分布估算：
        - 均值: 上车时间
        - 标准差: 时间差的一半（如果提供）
        
        Args:
            boarding_times: 上车时间数组
            time_diff: 时间差数组（可选）
            
        Returns:
            估算的到达时间数组
        """
        if time_diff is None:
            # 默认标准差为5分钟
            std_dev = 5.0
        else:
            std_dev = time_diff / 2.0
        
        # 使用正态分布生成到达时间
        arrival_times = np.random.normal(boarding_times, std_dev)
        
        # 确保到达时间不晚于上车时间
        arrival_times = np.minimum(arrival_times, boarding_times)
        
        # 确保到达时间为正数
        arrival_times = np.maximum(arrival_times, 0)
        
        return arrival_times
    
    def generate_flow_matrix(self, route: str, direction: int, 
                            time_resolution: int = 1) -> np.ndarray:
        """
        生成客流矩阵
        
        Args:
            route: 路线编号
            direction: 方向 (0或1)
            time_resolution: 时间分辨率（分钟）
            
        Returns:
            客流矩阵 [时间步, 站点]
        """
        if route not in self.passenger_data:
            raise ValueError(f"路线 {route} 的数据未加载")
        
        df = self.passenger_data[route][f'direction_{direction}']
        
        # 获取最大站点数
        max_station = max(df['Boarding station'].max(), 
                         df['Alighting station'].max()) + 1
        
        # 获取时间范围
        max_time = df['Boarding time'].max()
        time_steps = int(max_time / time_resolution) + 1
        
        # 初始化客流矩阵
        flow_matrix = np.zeros((time_steps, max_station))
        
        # 填充客流矩阵
        for _, row in df.iterrows():
            time_idx = int(row['Boarding time'] / time_resolution)
            station = int(row['Boarding station'])
            if time_idx < time_steps and station < max_station:
                flow_matrix[time_idx, station] += 1
        
        logger.info(f"路线{route}方向{direction}客流矩阵生成完成: "
                   f"形状 {flow_matrix.shape}, 总乘客数 {flow_matrix.sum():.0f}")
        
        return flow_matrix
    
    def get_statistics(self, route: str, direction: int) -> Dict:
        """
        获取数据统计信息
        
        Args:
            route: 路线编号
            direction: 方向 (0或1)
            
        Returns:
            统计信息字典
        """
        if route not in self.passenger_data:
            raise ValueError(f"路线 {route} 的数据未加载")
        
        df = self.passenger_data[route][f'direction_{direction}']
        
        stats = {
            'total_passengers': len(df),
            'unique_passengers': df['Label'].nunique(),
            'num_stations': max(df['Boarding station'].max(), 
                              df['Alighting station'].max()) + 1,
            'time_range': {
                'min': df['Boarding time'].min(),
                'max': df['Boarding time'].max(),
                'span_hours': (df['Boarding time'].max() - 
                             df['Boarding time'].min()) / 60
            },
            'boarding_stations': {
                'min': df['Boarding station'].min(),
                'max': df['Boarding station'].max(),
                'unique': df['Boarding station'].nunique()
            },
            'alighting_stations': {
                'min': df['Alighting station'].min(),
                'max': df['Alighting station'].max(),
                'unique': df['Alighting station'].nunique()
            }
        }
        
        return stats
    
    def get_hourly_distribution(self, route: str, direction: int) -> pd.Series:
        """
        获取小时级客流分布
        
        Args:
            route: 路线编号
            direction: 方向 (0或1)
            
        Returns:
            小时级客流分布
        """
        if route not in self.passenger_data:
            raise ValueError(f"路线 {route} 的数据未加载")
        
        df = self.passenger_data[route][f'direction_{direction}']
        df['hour'] = df['Boarding time'] // 60
        
        return df.groupby('hour').size()
