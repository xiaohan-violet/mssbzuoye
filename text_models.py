import torch
import torch.nn as nn
from transformers import BertModel
from config import MODEL_CONFIG


class TextModel(nn.Module):
    """基于BERT的文本情感分析模型"""

    def __init__(self, num_classes=None):
        super(TextModel, self).__init__()
        self.num_classes = num_classes or MODEL_CONFIG['text']['num_classes']
        # 加载预训练的BERT模型
        self.bert = BertModel.from_pretrained(MODEL_CONFIG['text']['bert_model'])
        # 防止过拟合的 dropout 层，随机"关闭"一部分神经元，防止模型过拟合（记住训练数据但不会泛化）
        self.dropout = nn.Dropout(0.3)
        # 分类器：将BERT特征映射到3个情感类别
        self.classifier = nn.Linear(
            MODEL_CONFIG['text']['hidden_size'],
            self.num_classes
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask# 指示哪些词是真实的，哪些是填充的
        )
        # 2. 获取BERT的[CLS]标记的输出（代表整个句子的含义）
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        # 4. 通过分类器得到3个类别的分数
        logits = self.classifier(pooled_output)
        return logits


class TextCNNModel(nn.Module):
    """BERT + TextCNN 文本情感分析模型"""

    def __init__(self, num_classes=None, num_filters=100, filter_sizes=[3, 4, 5]):
        super(TextCNNModel, self).__init__()
        self.num_classes = num_classes or MODEL_CONFIG['text']['num_classes']

        self.bert = BertModel.from_pretrained(MODEL_CONFIG['text']['bert_model'])
        # 创建多个不同大小的卷积核
        # 不同大小的卷积核捕捉不同范围的情感，更加综合
        self.convs = nn.ModuleList([
            nn.Conv1d(
                MODEL_CONFIG['text']['hidden_size'],
                num_filters,
                kernel_size=fs
            )
            for fs in filter_sizes
        ])

        self.dropout = nn.Dropout(0.3)
        # 分类器：输入维度 = 卷积核数量 × 每个卷积核的输出通道数
        self.classifier = nn.Linear(
            num_filters * len(filter_sizes),
            self.num_classes
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # 获取所有token的表示
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        sequence_output = sequence_output.permute(0, 2, 1)  # [batch_size, hidden_size, seq_len]

        # 应用多个卷积核
        conv_outputs = []
        for conv in self.convs:
            conv_out = conv(sequence_output)  # [batch_size, num_filters, seq_len - filter_size + 1]
            conv_out = torch.relu(conv_out)
            pooled = torch.max(conv_out, dim=2)[0]  # [batch_size, num_filters]
            conv_outputs.append(pooled)

        # 拼接所有卷积结果
        combined = torch.cat(conv_outputs, dim=1)  # [batch_size, num_filters * len(filter_sizes)]
        combined = self.dropout(combined)
        # 5. 最终分类
        logits = self.classifier(combined)

        return logits