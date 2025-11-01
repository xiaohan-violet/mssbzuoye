import torch
import torch.nn as nn
from config import MODEL_CONFIG

# 1. MTFN - 多任务融合网络（特征级融合）
# 2. MLMF - 多模态低秩融合（压缩融合）
# 3. LateFusion - 晚期融合（决策级融合）
class MTFN(nn.Module):
    """多任务融合网络"""
#MTFN采用早期融合策略，在特征层面进行融合。每个模态先独立处理，然后将处理后的特征拼接起来，最后进行联合分类。
    def __init__(self, text_dim=None, audio_dim=None, video_dim=None,
                 num_classes=None, hidden_size=128):
        super(MTFN, self).__init__()
        self.text_dim = text_dim or MODEL_CONFIG['fusion']['text_dim']
        self.audio_dim = audio_dim or MODEL_CONFIG['fusion']['audio_dim']
        self.video_dim = video_dim or MODEL_CONFIG['fusion']['video_dim']
        self.num_classes = num_classes or MODEL_CONFIG['fusion']['num_classes']

        # 文本分支
        self.text_fc = nn.Linear(self.text_dim, hidden_size)
        self.text_bn = nn.BatchNorm1d(hidden_size)

        # 音频分支
        self.audio_fc = nn.Linear(self.audio_dim, hidden_size)
        self.audio_bn = nn.BatchNorm1d(hidden_size)

        # 视频分支
        self.video_fc = nn.Linear(self.video_dim, hidden_size)
        self.video_bn = nn.BatchNorm1d(hidden_size)

        # 融合层
        self.fusion_fc = nn.Linear(hidden_size * 3, hidden_size)
        self.fusion_bn = nn.BatchNorm1d(hidden_size)

        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_size, self.num_classes)

    def forward(self, text_features, audio_features, video_features):
        # 处理文本特征
        text_out = torch.relu(self.text_bn(self.text_fc(text_features)))

        # 处理音频特征
        audio_out = torch.relu(self.audio_bn(self.audio_fc(audio_features)))

        # 处理视频特征
        video_out = torch.relu(self.video_bn(self.video_fc(video_features)))

        # 特征融合
        fused = torch.cat([text_out, audio_out, video_out], dim=1)
        fused = torch.relu(self.fusion_bn(self.fusion_fc(fused)))
        fused = self.dropout(fused)

        logits = self.classifier(fused)
        return logits


class MLMF(nn.Module):
    """多模态低秩融合方法"""
#MLMF采用低秩融合策略，先将每个模态的特征投影到低维空间，然后进行融合，这样可以减少参数数量，防止过拟合。
    def __init__(self, text_dim=None, audio_dim=None, video_dim=None,
                 num_classes=None, hidden_size=128, rank=16):
        super(MLMF, self).__init__()
        self.text_dim = text_dim or MODEL_CONFIG['fusion']['text_dim']
        self.audio_dim = audio_dim or MODEL_CONFIG['fusion']['audio_dim']
        self.video_dim = video_dim or MODEL_CONFIG['fusion']['video_dim']
        self.num_classes = num_classes or MODEL_CONFIG['fusion']['num_classes']
        self.rank = rank

        # 低秩投影矩阵
        self.text_proj = nn.Linear(self.text_dim, self.rank, bias=False)
        self.audio_proj = nn.Linear(self.audio_dim, self.rank, bias=False)
        self.video_proj = nn.Linear(self.video_dim, self.rank, bias=False)

        # 模态特定权重
        self.text_weight = nn.Parameter(torch.ones(1))
        self.audio_weight = nn.Parameter(torch.ones(1))
        self.video_weight = nn.Parameter(torch.ones(1))

        # 融合层
        self.fusion_fc = nn.Linear(self.rank * 3, hidden_size)
        self.fusion_bn = nn.BatchNorm1d(hidden_size)

        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_size, self.num_classes)

    def forward(self, text_features, audio_features, video_features):
        # 低秩投影
        text_low_rank = self.text_proj(text_features) * self.text_weight
        audio_low_rank = self.audio_proj(audio_features) * self.audio_weight
        video_low_rank = self.video_proj(video_features) * self.video_weight

        # 特征融合
        fused = torch.cat([text_low_rank, audio_low_rank, video_low_rank], dim=1)
        fused = torch.relu(self.fusion_bn(self.fusion_fc(fused)))
        fused = self.dropout(fused)

        logits = self.classifier(fused)
        return logits


class LateFusion(nn.Module):
    """晚期融合模型"""
#LateFusion采用决策级融合，每个模态先独立进行分类，然后将分类结果进行加权融合得到最终决策。灵感来自于MOE
    def __init__(self, text_dim=None, audio_dim=None, video_dim=None, num_classes=None):
        super(LateFusion, self).__init__()
        self.text_dim = text_dim or MODEL_CONFIG['fusion']['text_dim']
        self.audio_dim = audio_dim or MODEL_CONFIG['fusion']['audio_dim']
        self.video_dim = video_dim or MODEL_CONFIG['fusion']['video_dim']
        self.num_classes = num_classes or MODEL_CONFIG['fusion']['num_classes']

        # 各模态分类器
        self.text_classifier = nn.Linear(self.text_dim, self.num_classes)
        self.audio_classifier = nn.Linear(self.audio_dim, self.num_classes)
        self.video_classifier = nn.Linear(self.video_dim, self.num_classes)

        # 融合权重
        self.text_weight = nn.Parameter(torch.ones(1))
        self.audio_weight = nn.Parameter(torch.ones(1))
        self.video_weight = nn.Parameter(torch.ones(1))

    def forward(self, text_features, audio_features, video_features):
        # 各模态预测
        text_logits = self.text_classifier(text_features)
        audio_logits = self.audio_classifier(audio_features)
        video_logits = self.video_classifier(video_features)

        # 加权融合
        fused_logits = (
                self.text_weight * text_logits +
                self.audio_weight * audio_logits +
                self.video_weight * video_logits
        )

        return fused_logits