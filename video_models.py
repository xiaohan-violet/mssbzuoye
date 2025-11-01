import torch
import torch.nn as nn
from config import MODEL_CONFIG


class VideoModel(nn.Module):
    """视频特征处理模型"""

    def __init__(self, input_dim=None, num_classes=None, hidden_size=256):
        super(VideoModel, self).__init__()
        self.input_dim = input_dim or MODEL_CONFIG['video']['input_dim']
        self.num_classes = num_classes or MODEL_CONFIG['video']['num_classes']

        self.fc1 = nn.Linear(self.input_dim, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)

        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)

        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_size // 2, self.num_classes)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)

        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)

        logits = self.classifier(x)
        return logits


class VideoLSTM(nn.Module):
    """视频时序特征处理模型"""

    def __init__(self, input_dim=None, num_classes=None, hidden_size=128, num_layers=2):
        super(VideoLSTM, self).__init__()
        self.input_dim = input_dim or MODEL_CONFIG['video']['input_dim']
        self.num_classes = num_classes or MODEL_CONFIG['video']['num_classes']

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_size * 2, self.num_classes)

    def forward(self, x):
        # x: [batch_size, seq_length, input_dim]
        lstm_out, (h_n, c_n) = self.lstm(x)

        # 使用最后一个时间步的输出
        output = lstm_out[:, -1, :]
        output = self.dropout(output)

        logits = self.classifier(output)
        return logits