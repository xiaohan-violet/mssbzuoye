import torch
import torch.nn as nn
from config import MODEL_CONFIG


class AudioFCN(nn.Module):
    """全连接网络用于音频静态特征"""

    def __init__(self, input_dim=None, num_classes=None, hidden_sizes=[256, 128, 64]):
        super(AudioFCN, self).__init__()
        self.input_dim = input_dim or MODEL_CONFIG['audio']['input_dim']
        self.num_classes = num_classes or MODEL_CONFIG['audio']['num_classes']

        # 构建全连接层
        layers = []
        prev_size = self.input_dim

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_size = hidden_size

        # 输出层（没有激活函数，因为后面用交叉熵损失）
        layers.append(nn.Linear(prev_size, self.num_classes))
        # 将所有层组合成顺序模型
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x: [batch_size, input_dim] - 确保输入是float
        if x.dtype != torch.float32:
            x = x.float()
        return self.network(x)


class AudioSimple(nn.Module):
    """简单的音频分类器"""

    def __init__(self, input_dim=None, num_classes=None, hidden_size=128):
        super(AudioSimple, self).__init__()
        self.input_dim = input_dim or MODEL_CONFIG['audio']['input_dim']
        self.num_classes = num_classes or MODEL_CONFIG['audio']['num_classes']

        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, self.num_classes)
        )

    def forward(self, x):
        # 确保输入是float
        if x.dtype != torch.float32:
            x = x.float()
        return self.classifier(x)


# 其他音频模型也添加数据类型检查
#即使我们只有静态的音频特征，也把它当作时序数据来处理，模拟音频在时间上的变化。
class TIMNET(nn.Module):
    """时间感知双向多尺度网络 - 适配静态特征"""

    def __init__(self, input_dim=None, num_classes=None, hidden_size=64):
        super(TIMNET, self).__init__()
        self.input_dim = input_dim or MODEL_CONFIG['audio']['input_dim']
        self.num_classes = num_classes or MODEL_CONFIG['audio']['num_classes']

        # 将静态特征转换为伪时序特征
        self.feature_expansion = nn.Sequential(
            nn.Linear(self.input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU()
        )

        # 时序处理层，门控循环单元，擅长处理序列数据，能记住之前的信息
        self.gru = nn.GRU(
            hidden_size * 2,
            hidden_size,
            batch_first=True,
            bidirectional=True,
            num_layers=2
        )

        self.classifier = nn.Linear(hidden_size * 2, self.num_classes)

    def forward(self, x):
        # x: [batch_size, input_dim] - 确保输入是float
        if x.dtype != torch.float32:
            x = x.float()

        batch_size = x.size(0)

        # 扩展特征维度
        expanded = self.feature_expansion(x)  # [batch_size, hidden_size*2]

        # 重塑为伪时序数据 [batch_size, seq_len=1, features]
        pseudo_temporal = expanded.unsqueeze(1)

        # 重复创建序列 [batch_size, seq_len=10, features]
        seq_features = pseudo_temporal.repeat(1, 10, 1)

        # GRU处理
        gru_out, _ = self.gru(seq_features)

        # 取最后一个时间步
        output = gru_out[:, -1, :]

        logits = self.classifier(output)
        return logits


class CAMPlusPlus(nn.Module):
    """改进的通道注意力机制网络 - 适配静态特征"""

    def __init__(self, input_dim=None, num_classes=None, hidden_size=128):
        super(CAMPlusPlus, self).__init__()
        self.input_dim = input_dim or MODEL_CONFIG['audio']['input_dim']
        self.num_classes = num_classes or MODEL_CONFIG['audio']['num_classes']

        # 特征变换
        self.feature_transform = nn.Sequential(
            nn.Linear(self.input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 8),
            nn.ReLU(),
            nn.Linear(hidden_size // 8, hidden_size),
            nn.Sigmoid()
        )

        self.classifier = nn.Linear(hidden_size, self.num_classes)

    def forward(self, x):
        # x: [batch_size, input_dim] - 确保输入是float
        if x.dtype != torch.float32:
            x = x.float()

        transformed = self.feature_transform(x)

        # 通道注意力
        attention_weights = self.channel_attention(transformed)
        attended = transformed * attention_weights

        logits = self.classifier(attended)
        return logits

class CAMPlusPlus(nn.Module):
    """改进的通道注意力机制网络 - 适配静态特征"""

    def __init__(self, input_dim=None, num_classes=None, hidden_size=128):
        super(CAMPlusPlus, self).__init__()
        self.input_dim = input_dim or MODEL_CONFIG['audio']['input_dim']
        self.num_classes = num_classes or MODEL_CONFIG['audio']['num_classes']

        # 特征变换
        self.feature_transform = nn.Sequential(
            nn.Linear(self.input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 8),
            nn.ReLU(),
            nn.Linear(hidden_size // 8, hidden_size),
            nn.Sigmoid()
        )

        self.classifier = nn.Linear(hidden_size, self.num_classes)

    def forward(self, x):
        # x: [batch_size, input_dim]
        transformed = self.feature_transform(x)

        # 通道注意力
        attention_weights = self.channel_attention(transformed)
        attended = transformed * attention_weights

        logits = self.classifier(attended)
        return logits