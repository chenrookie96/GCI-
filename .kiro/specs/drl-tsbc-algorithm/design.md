# DRL-TSBC 算法复现设计文档

## 概述

本文档详细设计DRL-TSBC算法的实现架构。系统采用模块化设计，主要包括数据处理模块、仿真环境模块、DQN网络模块、训练模块、推理模块和可视化模块。整体架构基于PyTorch深度学习框架，支持GPU加速训练。

## 系统架构

### 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                        主程序入口                             │
│                    (main.py / train.py)                      │
└────────────┬────────────────────────────────────────────────┘
             │
             ├──────────────┬──────────────┬──────────────────┐
             │              │              │                  │
        ┌────▼────┐    ┌────▼────┐   ┌────▼────┐      ┌──────▼──────┐
        │数据模块  │    │仿真环境  │   │DQN模块  │      │可视化模块    │
        └────┬────┘    └────┬────┘   └────┬────┘      └──────┬──────┘
             │              │              │                  │
             │              │              │                  │
        ┌────▼──────────────▼──────────────▼──────────────────▼────┐
        │                    工具和配置模块                          │
        │         (config.py / utils.py / metrics.py)              │
        └──────────────────────────────────────────────────────────┘
```

### 技术栈

- Python 3.8+
- PyTorch 2.0+ (支持CUDA 11.8)
- NumPy 1.24+
- Pandas 2.0+
- Matplotlib 3.7+
- Seaborn 0.12+

## 核心组件设计

### 1. 数据处理模块 (data_loader.py)


#### 职责
- 加载和解析CSV格式的乘客数据和交通数据
- 数据预处理和验证
- 提供数据访问接口

#### 类设计

**PassengerDataLoader**
```python
class PassengerDataLoader:
    def __init__(self, route_id: int, direction: int):
        """
        参数:
            route_id: 线路编号 (208, 211, 683)
            direction: 方向 (0=上行, 1=下行)
        """
        
    def load_passenger_data(self) -> pd.DataFrame:
        """加载乘客数据，返回包含以下列的DataFrame:
        - Label: 乘客唯一标识
        - Boarding time: 上车时间(分钟)
        - Boarding station: 上车站点
        - Alighting station: 下车站点
        - Arrival time: 到达站点时间(分钟)
        """
        
    def load_traffic_data(self) -> pd.DataFrame:
        """加载交通数据，返回包含站间行驶时间的DataFrame"""
        
    def get_passengers_at_time(self, time_minute: int) -> List[Passenger]:
        """获取指定时间到达的乘客列表"""
```

**TrafficDataLoader**
```python
class TrafficDataLoader:
    def __init__(self, route_id: int, direction: int):
        pass
        
    def get_travel_time(self, from_station: int, to_station: int, 
                       current_time: int) -> int:
        """获取站间行驶时间(分钟)"""
```

#### 数据结构

**Passenger**
```python
@dataclass
class Passenger:
    label: str
    arrival_time: int  # 到达站点的时间(分钟)
    boarding_time: int  # 实际上车时间(分钟)
    boarding_station: int
    alighting_station: int
    waiting_time: int = 0  # 等待时间
    is_stranded: bool = False  # 是否滞留
```

### 2. 公交仿真环境模块 (bus_env.py)

#### 职责
- 模拟公交运营环境
- 处理车辆运行、乘客上下车
- 计算状态向量和奖励
- 提供标准的强化学习环境接口

#### 类设计

**BusEnvironment**
```python
class BusEnvironment:
    def __init__(self, config: EnvConfig):
        """
        参数:
            config: 环境配置，包含线路信息、参数等
        """
        self.route_id = config.route_id
        self.direction = config.direction
        self.num_stations = config.num_stations
        self.start_time = config.start_time  # 首班车时间(分钟)
        self.end_time = config.end_time  # 末班车时间(分钟)
        self.max_capacity = config.max_capacity  # 最大载客量
        self.seats = config.seats  # 座位数
        self.alpha = 1.5  # 站立系数
        
        # 奖励函数参数
        self.omega = config.omega  # 等待时间惩罚权重
        self.beta = 0.2  # 滞留乘客惩罚权重
        self.zeta = 0.002  # 发车平衡权重
        self.mu = 5000  # 等待时间归一化参数
        self.delta = 200  # 发车次数归一化参数
        
        # 发车约束
        self.t_min = config.t_min  # 最小发车间隔
        self.t_max = config.t_max  # 最大发车间隔
        
        # 状态变量
        self.current_time = self.start_time
        self.buses = []  # 运行中的公交车列表
        self.dispatch_count_up = 0  # 上行发车次数
        self.dispatch_count_down = 0  # 下行发车次数
        self.last_dispatch_time_up = 0
        self.last_dispatch_time_down = 0
        
        # 乘客管理
        self.waiting_passengers = {i: [] for i in range(self.num_stations)}
        self.stranded_passengers = []
        
    def reset(self) -> np.ndarray:
        """重置环境，返回初始状态"""
        
    def step(self, action: Tuple[int, int]) -> Tuple[np.ndarray, float, bool, dict]:
        """
        执行一步仿真
        参数:
            action: (a_up, a_down) 上下行是否发车
        返回:
            state: 下一状态
            reward: 奖励值
            done: 是否结束
            info: 额外信息
        """
        
    def _get_state(self) -> np.ndarray:
        """计算当前状态向量"""
        
    def _calculate_reward(self, action: Tuple[int, int]) -> float:
        """计算奖励值"""
        
    def _dispatch_bus(self, direction: int):
        """发车，创建新的公交车实例"""
        
    def _update_buses(self):
        """更新所有运行中的公交车状态"""
        
    def _update_passengers(self):
        """更新乘客状态，添加新到达的乘客"""
```


**Bus**
```python
@dataclass
class Bus:
    bus_id: int
    direction: int  # 0=上行, 1=下行
    current_station: int
    passengers: List[Passenger]
    capacity: int
    dispatch_time: int
    
    def move_to_next_station(self, travel_time: int):
        """移动到下一站"""
        
    def board_passengers(self, waiting_passengers: List[Passenger]) -> List[Passenger]:
        """乘客上车，返回滞留乘客列表"""
        
    def alight_passengers(self) -> List[Passenger]:
        """乘客下车"""
        
    def get_load(self) -> int:
        """获取当前载客量"""
```

#### 状态空间计算

状态向量维度: 2 + 4 + 4 = 10维

```python
def _compute_state_vector(self) -> np.ndarray:
    """
    返回10维状态向量:
    [0-1]: 全局时间状态 (a_m^1, a_m^2)
    [2-5]: 上行状态 (x_m^1, x_m^2, x_m^3, x_m^4)
    [6-9]: 下行状态 (y_m^1, y_m^2, y_m^3, y_m^4)
    """
    # 时间状态
    hour = self.current_time // 60
    minute = self.current_time % 60
    a1 = hour / 24.0
    a2 = minute / 60.0
    
    # 上行状态
    x1 = self._compute_max_load_ratio(direction=0)  # 满载率
    x2 = self._compute_waiting_time(direction=0) / self.mu  # 归一化等待时间
    x3 = self._compute_capacity_utilization(direction=0)  # 容量利用率
    x4 = (self.dispatch_count_up - self.dispatch_count_down) / self.delta
    
    # 下行状态
    y1 = self._compute_max_load_ratio(direction=1)
    y2 = self._compute_waiting_time(direction=1) / self.mu
    y3 = self._compute_capacity_utilization(direction=1)
    y4 = (self.dispatch_count_down - self.dispatch_count_up) / self.delta
    
    return np.array([a1, a2, x1, x2, x3, x4, y1, y2, y3, y4], dtype=np.float32)
```

#### 奖励函数计算

```python
def _compute_reward(self, action: Tuple[int, int]) -> float:
    """
    计算总奖励 = 上行奖励 + 下行奖励
    """
    a_up, a_down = action
    
    # 上行奖励
    o_up = self._compute_actual_capacity(direction=0)
    e_up = self._compute_provided_capacity(direction=0)
    W_up = self._compute_total_waiting_time(direction=0)
    d_up = self._count_stranded_passengers(direction=0)
    
    if a_up == 0:  # 不发车
        r_up = 1 - (o_up / e_up) - (self.omega * W_up) - (self.beta * d_up) + \
               self.zeta * (self.dispatch_count_up - self.dispatch_count_down)
    else:  # 发车
        r_up = (o_up / e_up) - (self.beta * d_up) - \
               self.zeta * (self.dispatch_count_up - self.dispatch_count_down)
    
    # 下行奖励 (类似计算)
    o_down = self._compute_actual_capacity(direction=1)
    e_down = self._compute_provided_capacity(direction=1)
    W_down = self._compute_total_waiting_time(direction=1)
    d_down = self._count_stranded_passengers(direction=1)
    
    if a_down == 0:
        r_down = 1 - (o_down / e_down) - (self.omega * W_down) - (self.beta * d_down) - \
                 self.zeta * (self.dispatch_count_up - self.dispatch_count_down)
    else:
        r_down = (o_down / e_down) - (self.beta * d_down) + \
                 self.zeta * (self.dispatch_count_up - self.dispatch_count_down)
    
    return r_up + r_down
```

### 3. DQN网络模块 (dqn_network.py)

#### 职责
- 定义深度Q网络架构
- 实现前向传播
- 支持GPU加速

#### 网络架构

```python
class DQN(nn.Module):
    def __init__(self, state_dim: int = 10, action_dim: int = 4, 
                 hidden_dim: int = 500, num_layers: int = 12):
        """
        参数:
            state_dim: 状态维度 (默认10)
            action_dim: 动作维度 (默认4: (0,0), (0,1), (1,0), (1,1))
            hidden_dim: 隐藏层神经元数量 (默认500)
            num_layers: 隐藏层数量 (默认12)
        """
        super(DQN, self).__init__()
        
        layers = []
        # 输入层
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        # 隐藏层
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        # 输出层
        layers.append(nn.Linear(hidden_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """使用正态分布初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.1)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        参数:
            state: 状态张量 [batch_size, state_dim]
        返回:
            Q值 [batch_size, action_dim]
        """
        return self.network(state)
```


### 4. 经验回放模块 (replay_buffer.py)

#### 职责
- 存储和管理训练经验
- 提供随机采样接口

#### 类设计

```python
class ReplayBuffer:
    def __init__(self, capacity: int = 3000):
        """
        参数:
            capacity: 经验池容量
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """添加经验"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """随机采样一批经验"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self) -> int:
        return len(self.buffer)
```

### 5. DQN训练模块 (dqn_agent.py)

#### 职责
- 实现DQN训练算法
- 管理主网络和目标网络
- 实现ε-贪婪策略
- 执行训练循环

#### 类设计

```python
class DQNAgent:
    def __init__(self, config: AgentConfig):
        """
        参数:
            config: 智能体配置
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 网络
        self.policy_net = DQN(state_dim=10, action_dim=4).to(self.device)
        self.target_net = DQN(state_dim=10, action_dim=4).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        
        # 超参数
        self.gamma = 0.4  # 折扣因子
        self.epsilon = 0.1  # 探索率
        self.batch_size = 64
        self.learning_freq = 5  # 学习频率
        self.target_update_freq = 100  # 目标网络更新频率
        
        # 经验回放
        self.replay_buffer = ReplayBuffer(capacity=3000)
        
        # 训练统计
        self.steps = 0
        self.learn_steps = 0
        
    def select_action(self, state: np.ndarray, epsilon: float = None) -> int:
        """
        选择动作 (ε-贪婪策略)
        返回动作索引: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        """
        if epsilon is None:
            epsilon = self.epsilon
            
        if random.random() < epsilon:
            return random.randint(0, 3)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
    
    def action_index_to_tuple(self, action_idx: int) -> Tuple[int, int]:
        """将动作索引转换为元组"""
        actions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        return actions[action_idx]
    
    def learn(self):
        """从经验池采样并更新网络"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # 采样
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # 转换为张量
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 计算当前Q值
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # 计算损失
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.learn_steps += 1
        
        # 更新目标网络
        if self.learn_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def train(self, env: BusEnvironment, num_episodes: int = 50):
        """
        训练循环
        """
        episode_rewards = []
        episode_dispatch_counts = []
        
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # 选择动作
                action_idx = self.select_action(state)
                action = self.action_index_to_tuple(action_idx)
                
                # 执行动作
                next_state, reward, done, info = env.step(action)
                
                # 存储经验
                self.replay_buffer.push(state, action_idx, reward, next_state, done)
                
                # 学习
                if self.steps % self.learning_freq == 0 and len(self.replay_buffer) >= self.batch_size:
                    self.learn()
                
                state = next_state
                episode_reward += reward
                self.steps += 1
            
            episode_rewards.append(episode_reward)
            episode_dispatch_counts.append({
                'up': env.dispatch_count_up,
                'down': env.dispatch_count_down
            })
            
            print(f"Episode {episode+1}/{num_episodes}, "
                  f"Reward: {episode_reward:.2f}, "
                  f"Dispatch: Up={env.dispatch_count_up}, Down={env.dispatch_count_down}")
        
        return episode_rewards, episode_dispatch_counts
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
```


### 6. 推理模块 (inference.py)

#### 职责
- 使用训练好的模型生成时刻表
- 实现时刻表后处理逻辑
- 确保上下行发车次数平衡

#### 类设计

```python
class ScheduleGenerator:
    def __init__(self, agent: DQNAgent, env: BusEnvironment):
        self.agent = agent
        self.env = env
    
    def generate_schedule(self) -> Dict[str, List[int]]:
        """
        生成公交时刻表
        返回: {'up': [发车时间列表], 'down': [发车时间列表]}
        """
        state = self.env.reset()
        schedule_up = []
        schedule_down = []
        done = False
        
        while not done:
            # 使用贪婪策略选择动作
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
        
        # 后处理：平衡上下行发车次数
        schedule_up, schedule_down = self._balance_schedules(schedule_up, schedule_down)
        
        return {'up': schedule_up, 'down': schedule_down}
    
    def _balance_schedules(self, schedule_up: List[int], 
                          schedule_down: List[int]) -> Tuple[List[int], List[int]]:
        """
        平衡上下行发车次数
        算法：
        1. 找到发车次数较多的方向
        2. 删除该方向的倒数第二次发车
        3. 从后向前调整发车时间，确保间隔不超过T_max
        """
        if len(schedule_up) == len(schedule_down):
            return schedule_up, schedule_down
        
        # 确定需要调整的方向
        if len(schedule_up) > len(schedule_down):
            schedule = schedule_up
        else:
            schedule = schedule_down
        
        # 删除倒数第二次发车
        if len(schedule) >= 2:
            schedule.pop(-2)
        
        # 向前调整发车时间
        k = len(schedule) - 1
        while k > 0:
            if schedule[k] - schedule[k-1] > self.env.t_max:
                schedule[k-1] = schedule[k] - self.env.t_max
            k -= 1
        
        if len(schedule_up) > len(schedule_down):
            return schedule, schedule_down
        else:
            return schedule_up, schedule
```

### 7. 评估模块 (evaluation.py)

#### 职责
- 评估时刻表性能
- 计算关键指标
- 支持对比实验

#### 类设计

```python
class ScheduleEvaluator:
    def __init__(self, env: BusEnvironment):
        self.env = env
    
    def evaluate_schedule(self, schedule: Dict[str, List[int]]) -> Dict:
        """
        评估时刻表性能
        返回指标字典
        """
        # 重置环境并应用时刻表
        state = self.env.reset()
        
        total_waiting_time_up = 0
        total_waiting_time_down = 0
        total_passengers_up = 0
        total_passengers_down = 0
        stranded_passengers_up = 0
        stranded_passengers_down = 0
        
        # 模拟运行
        for time in range(self.env.start_time, self.env.end_time + 1):
            # 根据时刻表发车
            action = (
                1 if time in schedule['up'] else 0,
                1 if time in schedule['down'] else 0
            )
            
            next_state, reward, done, info = self.env.step(action)
            
            # 收集统计数据
            total_waiting_time_up += info.get('waiting_time_up', 0)
            total_waiting_time_down += info.get('waiting_time_down', 0)
            total_passengers_up += info.get('passengers_up', 0)
            total_passengers_down += info.get('passengers_down', 0)
            stranded_passengers_up += info.get('stranded_up', 0)
            stranded_passengers_down += info.get('stranded_down', 0)
        
        # 计算平均等待时间
        avg_waiting_time_up = total_waiting_time_up / max(total_passengers_up, 1)
        avg_waiting_time_down = total_waiting_time_down / max(total_passengers_down, 1)
        
        return {
            'dispatch_count_up': len(schedule['up']),
            'dispatch_count_down': len(schedule['down']),
            'avg_waiting_time_up': avg_waiting_time_up,
            'avg_waiting_time_down': avg_waiting_time_down,
            'stranded_passengers_up': stranded_passengers_up,
            'stranded_passengers_down': stranded_passengers_down,
            'total_passengers_up': total_passengers_up,
            'total_passengers_down': total_passengers_down
        }
    
    def compute_capacity_over_time(self, schedule: Dict[str, List[int]], 
                                   time_interval: int = 30) -> pd.DataFrame:
        """
        计算每个时间段的客运容量
        用于生成可视化图表
        """
        # 实现逻辑...
        pass
```

### 8. 可视化模块 (visualization.py)

#### 职责
- 生成训练曲线图
- 生成性能对比图
- 生成参数敏感性分析图
- 生成客流适应性测试图


#### 类设计

```python
class Visualizer:
    def __init__(self, output_dir: str = 'results/figures'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_training_convergence(self, episode_rewards: List[float],
                                  dispatch_counts: List[Dict],
                                  title: str = 'DRL-TSBC 收敛结果',
                                  filename: str = 'convergence.png'):
        """
        绘制训练收敛曲线 (图2-11)
        双Y轴：左侧为平均奖励，右侧为发车次数
        """
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        episodes = range(1, len(episode_rewards) + 1)
        
        # 左Y轴：平均奖励
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('平均奖励', color='blue')
        ax1.plot(episodes, episode_rewards, color='blue', label='平均奖励')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True, alpha=0.3)
        
        # 右Y轴：发车次数
        ax2 = ax1.twinx()
        ax2.set_ylabel('发车次数', color='orange')
        
        dispatch_up = [d['up'] for d in dispatch_counts]
        dispatch_down = [d['down'] for d in dispatch_counts]
        
        ax2.plot(episodes, dispatch_up, color='orange', label='上行发车次数')
        ax2.plot(episodes, dispatch_down, color='purple', label='下行发车次数')
        ax2.tick_params(axis='y', labelcolor='orange')
        
        # 图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()
    
    def plot_omega_sensitivity(self, omega_values: List[float],
                              results: List[Dict],
                              route_id: int,
                              filename: str = 'omega_sensitivity.png'):
        """
        绘制ω参数敏感性分析图 (图2-8)
        """
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # 提取数据
        dispatch_counts = [r['dispatch_count_up'] + r['dispatch_count_down'] for r in results]
        awt_up = [r['avg_waiting_time_up'] for r in results]
        awt_down = [r['avg_waiting_time_down'] for r in results]
        
        # 左Y轴：发车次数
        ax1.set_xlabel('ω')
        ax1.set_ylabel('发车次数 (NDT)', color='blue')
        ax1.plot(omega_values, dispatch_counts, color='blue', marker='o', label='发车次数')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True, alpha=0.3)
        
        # 右Y轴：平均等待时间
        ax2 = ax1.twinx()
        ax2.set_ylabel('乘客平均等待时间 (分钟)', color='orange')
        ax2.plot(omega_values, awt_up, color='orange', marker='s', label='上行AWT')
        ax2.plot(omega_values, awt_down, color='purple', marker='^', label='下行AWT')
        ax2.tick_params(axis='y', labelcolor='orange')
        
        # 图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        plt.title(f'DRL-TSBC 在不同 ω 下的 {route_id} 线发车次数与乘客平均等待时间的对比')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()
    
    def plot_capacity_comparison(self, time_points: List[int],
                                capacity_data: Dict[str, List[float]],
                                title: str,
                                filename: str):
        """
        绘制客运容量对比图 (图2-3, 2-4, 2-5, 2-6, 2-7)
        """
        plt.figure(figsize=(12, 6))
        
        # 转换时间点为小时:分钟格式
        time_labels = [f"{t//60}:{t%60:02d}" for t in time_points]
        
        for label, capacity in capacity_data.items():
            plt.plot(time_points, capacity, label=label, linewidth=2)
        
        plt.xlabel('时间')
        plt.ylabel('总客运量')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()
    
    def generate_comparison_table(self, results: List[Dict],
                                 filename: str = 'comparison_table.csv'):
        """
        生成性能对比表格 (表2-3)
        """
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(self.output_dir, filename), index=False, encoding='utf-8-sig')
        return df
```

### 9. 配置管理模块 (config.py)

#### 职责
- 管理所有超参数和配置
- 提供配置加载和保存接口

```python
@dataclass
class RouteConfig:
    """线路配置"""
    route_id: int
    direction: int
    num_stations: int
    start_time: int  # 分钟
    end_time: int  # 分钟
    max_capacity: int = 100
    seats: int = 40
    t_min: int = 5  # 最小发车间隔(分钟)
    t_max: int = 20  # 最大发车间隔(分钟)
    omega: float = 1/1000  # 等待时间惩罚权重

@dataclass
class DQNConfig:
    """DQN配置"""
    state_dim: int = 10
    action_dim: int = 4
    hidden_dim: int = 500
    num_layers: int = 12
    learning_rate: float = 0.001
    gamma: float = 0.4
    epsilon: float = 0.1
    batch_size: int = 64
    replay_buffer_size: int = 3000
    learning_freq: int = 5
    target_update_freq: int = 100
    num_episodes: int = 50

@dataclass
class ExperimentConfig:
    """实验配置"""
    route_config: RouteConfig
    dqn_config: DQNConfig
    seed: int = 42
    device: str = 'cuda'
    save_dir: str = 'results/models'
    log_dir: str = 'results/logs'

class ConfigManager:
    @staticmethod
    def get_route_208_config(direction: int) -> RouteConfig:
        """获取208线路配置"""
        return RouteConfig(
            route_id=208,
            direction=direction,
            num_stations=26 if direction == 0 else 24,
            start_time=6*60,  # 6:00
            end_time=21*60,  # 21:00
            omega=1/1000
        )
    
    @staticmethod
    def get_route_211_config(direction: int) -> RouteConfig:
        """获取211线路配置"""
        return RouteConfig(
            route_id=211,
            direction=direction,
            num_stations=17 if direction == 0 else 11,
            start_time=6*60,
            end_time=22*60,
            omega=1/900
        )
    
    @staticmethod
    def get_default_dqn_config() -> DQNConfig:
        """获取默认DQN配置"""
        return DQNConfig()
    
    @staticmethod
    def save_config(config: ExperimentConfig, path: str):
        """保存配置到JSON文件"""
        import json
        from dataclasses import asdict
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(asdict(config), f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def load_config(path: str) -> ExperimentConfig:
        """从JSON文件加载配置"""
        import json
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return ExperimentConfig(**data)
```


## 数据模型

### 核心数据结构

```python
# 乘客数据
@dataclass
class Passenger:
    label: str
    arrival_time: int
    boarding_time: int
    boarding_station: int
    alighting_station: int
    waiting_time: int = 0
    is_stranded: bool = False

# 公交车数据
@dataclass
class Bus:
    bus_id: int
    direction: int
    current_station: int
    passengers: List[Passenger]
    capacity: int
    dispatch_time: int
    arrival_times: List[int]  # 到达各站点的时间

# 环境状态
@dataclass
class EnvironmentState:
    current_time: int
    buses: List[Bus]
    waiting_passengers: Dict[int, List[Passenger]]
    dispatch_count_up: int
    dispatch_count_down: int
    last_dispatch_time_up: int
    last_dispatch_time_down: int
    stranded_passengers: List[Passenger]

# 评估结果
@dataclass
class EvaluationResult:
    route_id: int
    direction: int
    method: str  # 'DRL-TSBC', '人工方案', 'DRL-TO'
    dispatch_count: int
    avg_waiting_time: float
    stranded_passengers: int
    omega: float
```

## 错误处理

### 异常类型

```python
class DataLoadError(Exception):
    """数据加载错误"""
    pass

class EnvironmentError(Exception):
    """环境运行错误"""
    pass

class ModelError(Exception):
    """模型相关错误"""
    pass
```

### 错误处理策略

1. 数据加载阶段
   - 验证文件存在性
   - 检查数据格式和完整性
   - 提供详细的错误信息

2. 训练阶段
   - 捕获CUDA内存错误，提示降低batch_size
   - 记录训练异常并保存检查点
   - 支持从检查点恢复训练

3. 推理阶段
   - 验证模型文件存在
   - 检查输入数据有效性
   - 处理边界情况

## 测试策略

### 单元测试

1. 数据加载模块测试
   - 测试CSV文件解析
   - 测试数据预处理
   - 测试边界情况

2. 环境模块测试
   - 测试状态计算
   - 测试奖励计算
   - 测试发车约束
   - 测试乘客上下车逻辑

3. DQN模块测试
   - 测试网络前向传播
   - 测试梯度计算
   - 测试权重初始化

### 集成测试

1. 端到端训练测试
   - 使用小规模数据集
   - 验证训练流程完整性
   - 检查输出格式

2. 推理测试
   - 验证时刻表生成
   - 验证后处理逻辑
   - 验证评估指标计算

### 性能测试

1. 训练速度测试
   - 测量每个episode的训练时间
   - 测量GPU利用率
   - 优化瓶颈

2. 内存使用测试
   - 监控GPU内存使用
   - 监控系统内存使用
   - 优化内存占用

## 项目结构

```
drl-tsbc/
├── data/
│   ├── __init__.py
│   ├── data_loader.py          # 数据加载模块
│   └── preprocessor.py         # 数据预处理
├── environment/
│   ├── __init__.py
│   ├── bus_env.py              # 公交仿真环境
│   └── entities.py             # 实体类(Bus, Passenger)
├── models/
│   ├── __init__.py
│   ├── dqn_network.py          # DQN网络
│   ├── dqn_agent.py            # DQN智能体
│   └── replay_buffer.py        # 经验回放池
├── inference/
│   ├── __init__.py
│   ├── schedule_generator.py   # 时刻表生成
│   └── evaluator.py            # 评估模块
├── visualization/
│   ├── __init__.py
│   └── visualizer.py           # 可视化模块
├── utils/
│   ├── __init__.py
│   ├── config.py               # 配置管理
│   ├── logger.py               # 日志模块
│   └── metrics.py              # 指标计算
├── experiments/
│   ├── __init__.py
│   ├── train.py                # 训练脚本
│   ├── evaluate.py             # 评估脚本
│   ├── ablation_study.py       # 消融实验
│   └── sensitivity_analysis.py # 参数敏感性分析
├── tests/
│   ├── test_data_loader.py
│   ├── test_environment.py
│   ├── test_dqn.py
│   └── test_integration.py
├── results/
│   ├── models/                 # 保存的模型
│   ├── logs/                   # 训练日志
│   ├── figures/                # 生成的图表
│   └── tables/                 # 生成的表格
├── configs/
│   ├── route_208.json          # 208线路配置
│   ├── route_211.json          # 211线路配置
│   └── default.json            # 默认配置
├── requirements.txt            # 依赖包
├── README.md                   # 项目说明
└── main.py                     # 主入口
```

## 依赖包

```txt
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
tensorboard>=2.13.0
```

## 开发流程

### 阶段1: 基础设施搭建
1. 创建项目结构
2. 实现配置管理模块
3. 实现日志模块
4. 设置GPU环境

### 阶段2: 数据处理
1. 实现数据加载模块
2. 实现数据预处理
3. 验证数据正确性

### 阶段3: 环境实现
1. 实现基础实体类(Passenger, Bus)
2. 实现公交仿真环境
3. 实现状态计算
4. 实现奖励函数
5. 测试环境逻辑

### 阶段4: DQN实现
1. 实现DQN网络
2. 实现经验回放池
3. 实现DQN智能体
4. 测试训练流程

### 阶段5: 训练和评估
1. 实现训练脚本
2. 实现推理模块
3. 实现评估模块
4. 进行初步训练测试

### 阶段6: 实验和可视化
1. 实现可视化模块
2. 进行完整训练
3. 进行对比实验
4. 进行消融实验
5. 进行参数敏感性分析
6. 生成所有图表和表格

### 阶段7: 验证和优化
1. 对比论文结果
2. 调整参数
3. 优化性能
4. 完善文档

## 关键技术决策

### 1. 框架选择
- 使用PyTorch作为深度学习框架，原因：
  - 良好的GPU支持
  - 灵活的动态计算图
  - 丰富的社区资源

### 2. 状态表示
- 使用归一化的连续值表示状态
- 所有特征归一化到[0,1]范围
- 便于神经网络学习

### 3. 动作编码
- 将4种动作组合编码为0-3的整数
- 简化Q值输出和动作选择

### 4. 经验回放
- 使用固定大小的deque实现
- 先进先出策略
- 随机采样打破时间相关性

### 5. 目标网络更新
- 使用硬更新策略
- 每100次学习更新一次
- 提高训练稳定性

### 6. GPU加速
- 自动检测CUDA可用性
- 批量处理提高效率
- 支持混合精度训练(可选)

## 性能优化建议

1. 数据加载优化
   - 预加载所有数据到内存
   - 使用缓存机制
   - 并行数据处理

2. 训练优化
   - 使用GPU加速
   - 调整batch_size平衡速度和内存
   - 使用梯度裁剪防止梯度爆炸

3. 内存优化
   - 及时释放不用的张量
   - 使用torch.no_grad()减少内存占用
   - 控制经验池大小

## 可扩展性设计

1. 支持新线路
   - 通过配置文件添加新线路
   - 无需修改核心代码

2. 支持新算法
   - 抽象环境接口
   - 易于替换DQN为其他算法

3. 支持新实验
   - 模块化实验脚本
   - 统一的评估接口

---

**设计文档版本**: 1.0  
**最后更新**: 2025年1月
