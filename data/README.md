# 数据集说明

## 真实数据集来源

根据相关论文，本项目算法基于以下真实数据集进行验证：

### 1. BUPTAIOC/Transportation 数据集
- **链接**: https://github.com/BUPTAIOC/Transportation
- **来源论文**: "A reinforcement learning-based approach for online bus scheduling"
- **内容**: 青岛市4条公交线路的真实运营数据
  - 线路59: 104个发车时间点，平均行程时间41分钟
  - 线路60: 120个发车时间点，平均行程时间55分钟  
  - 线路803: 154个发车时间点，平均行程时间47分钟
  - 线路85: 170个发车时间点，平均行程时间35分钟
- **用途**: RL-BSA公交调度算法验证

### 2. 厦门公交数据集
- **链接**: https://blog.csdn.net/qq_24791311?spm=1000.2115.3001.5343
- **来源论文**: "Deep Reinforcement Learning based dynamic optimization of bus timetable"
- **内容**: 厦门市2018年6月某日的公交刷卡记录
  - 线路2: 上行4968条记录，下行4515条记录，37/36个站点
  - 线路230: 上行7739条记录，下行6818条记录，33/33个站点
  - 线路239: 上行5896条记录，下行5802条记录，35/36个站点
- **用途**: DRL-TO时刻表优化算法验证

## 数据格式要求

### 乘客流量数据
```json
{
  "passenger_id": "string",
  "boarding_time": "timestamp", 
  "boarding_station": "int",
  "alighting_station": "int",
  "arrival_time": "timestamp"
}
```

### 站点信息数据
```json
{
  "station_id": "int",
  "station_name": "string",
  "coordinates": {"lat": "float", "lng": "float"},
  "distance_to_next": "float"
}
```

### 车辆参数
```json
{
  "vehicle_capacity": 47,
  "max_working_time_short": 8,
  "max_driving_time_short": 6.5,
  "max_working_time_long": 16,
  "max_driving_time_long": 13,
  "min_rest_time": 3
}
```

## 数据预处理

1. **时间标准化**: 将时间转换为分钟表示（如6:00 AM = 360分钟）
2. **客流估算**: 对于同一站点同一时间上车的乘客，假设到达时间均匀分布
3. **特征归一化**: 将所有状态特征归一化到[0,1]范围
4. **缺失值处理**: 使用历史平均值填充缺失的客流数据

## 使用说明

1. 下载真实数据集用于算法验证
2. 根据数据格式要求准备自己的数据
3. 运行预处理脚本生成训练数据
4. 开始模型训练和评估
