"""DQN网络架构"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """深度Q网络"""
    
    def __init__(self, state_dim: int = 10, action_dim: int = 4,
                 hidden_dim: int = 500, num_layers: int = 12):
        """
        初始化DQN网络
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dim: 隐藏层神经元数量
            num_layers: 隐藏层数量
        """
        super(DQN, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 构建网络层
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
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            state: 状态张量 [batch_size, state_dim]
            
        Returns:
            Q值 [batch_size, action_dim]
        """
        return self.network(state)


if __name__ == '__main__':
    # 测试网络
    print("测试DQN网络...")
    
    # 创建网络
    dqn = DQN(state_dim=10, action_dim=4, hidden_dim=500, num_layers=12)
    print(f"网络参数数量: {sum(p.numel() for p in dqn.parameters())}")
    
    # 测试前向传播
    batch_size = 64
    state = torch.randn(batch_size, 10)
    q_values = dqn(state)
    print(f"输入形状: {state.shape}")
    print(f"输出形状: {q_values.shape}")
    print(f"Q值范围: [{q_values.min().item():.4f}, {q_values.max().item():.4f}]")
    
    # 测试GPU
    if torch.cuda.is_available():
        print("\nGPU可用，测试GPU运行...")
        dqn_gpu = dqn.cuda()
        state_gpu = state.cuda()
        q_values_gpu = dqn_gpu(state_gpu)
        print(f"GPU输出形状: {q_values_gpu.shape}")
        print("GPU测试成功！")
    else:
        print("\nGPU不可用，仅使用CPU")
