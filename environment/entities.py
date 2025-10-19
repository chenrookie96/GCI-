"""实体类定义：乘客和公交车"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class Passenger:
    """乘客数据类"""
    label: str
    arrival_time: int  # 到达站点的时间(分钟)
    boarding_time: int  # 实际上车时间(分钟)，初始为-1表示未上车
    boarding_station: int
    alighting_station: int
    waiting_time: int = 0  # 等待时间(分钟)
    is_stranded: bool = False  # 是否滞留


@dataclass
class Bus:
    """公交车数据类"""
    bus_id: int
    direction: int  # 0=上行, 1=下行
    current_station: int
    passengers: List[Passenger] = field(default_factory=list)
    capacity: int = 100
    dispatch_time: int = 0
    current_time: int = 0
    
    def get_load(self) -> int:
        """获取当前载客量"""
        return len(self.passengers)
    
    def is_full(self) -> bool:
        """判断是否满载"""
        return self.get_load() >= self.capacity
    
    def board_passengers(self, waiting_passengers: List[Passenger], 
                        current_time: int) -> List[Passenger]:
        """
        乘客上车
        
        Args:
            waiting_passengers: 等待上车的乘客列表
            current_time: 当前时间
            
        Returns:
            滞留乘客列表（因满载无法上车）
        """
        stranded = []
        
        for passenger in waiting_passengers:
            if not self.is_full():
                # 乘客上车
                passenger.boarding_time = current_time
                passenger.waiting_time = current_time - passenger.arrival_time
                self.passengers.append(passenger)
            else:
                # 车辆满载，乘客滞留
                passenger.is_stranded = True
                stranded.append(passenger)
        
        return stranded
    
    def alight_passengers(self, current_station: int) -> List[Passenger]:
        """
        乘客下车
        
        Args:
            current_station: 当前站点
            
        Returns:
            下车的乘客列表
        """
        alighting = []
        remaining = []
        
        for passenger in self.passengers:
            if passenger.alighting_station == current_station:
                alighting.append(passenger)
            else:
                remaining.append(passenger)
        
        self.passengers = remaining
        return alighting
    
    def move_to_next_station(self, travel_time: int):
        """
        移动到下一站
        
        Args:
            travel_time: 到下一站的行驶时间(分钟)
        """
        self.current_time += travel_time
        self.current_station += 1
