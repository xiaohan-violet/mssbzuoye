import torch
import os

# 设置代理端口
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'

# 设备配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# 数据路径
DATA_PATH = {
    'metadata': 'data/meta.csv',
    'unaligned': 'data/unaligned.pkl',
    'raw_video': 'data/raw/',
    'processed': 'data/processed/'
}

# config.py - 更新特征维度
MODEL_CONFIG = {
    'text': {
        'bert_model': 'bert-base-chinese',
        'hidden_size': 768,
        'num_classes': 3,
        'max_length': 128
    },
    'audio': {
        'input_dim': 1024,  # 更新为Wav2Vec2特征维度
        'hidden_size': 256,  # 增加隐藏层大小
        'num_classes': 3
    },
    'video': {
        'input_dim': 2048,  # 更新为ResNet特征维度
        'hidden_size': 512,  # 增加隐藏层大小
        'num_classes': 3
    },
    'fusion': {
        'text_dim': 768,
        'audio_dim': 1024,   # 更新音频维度
        'video_dim': 2048,   # 更新视频维度
        'hidden_size': 256,
        'num_classes': 3
    }
}

# 改进的训练参数
TRAIN_CONFIG = {
    'batch_size': 32,
    'learning_rate': 2e-5,  # 更小的学习率
    'epochs': 100,          # 更多epochs
    'patience': 15,         # 更长的耐心
    'weight_decay': 1e-4,   # 更强的正则化
    'use_augmentation': True,
    'use_class_balancing': True,
    'use_focal_loss': True
}

# 数据增强参数
AUGMENT_CONFIG = {
    'text_aug_prob': 0.3,
    'audio_aug_prob': 0.3,
    'video_aug_prob': 0.3,
    'noise_std': 0.01,
    'scale_range': [0.9, 1.1],
    'shift_range': [-0.1, 0.1]
}

# 特征提取参数
FEATURE_CONFIG = {
    'audio': {
        'sr': 16000,
        'n_mfcc': 13,
        'n_fft': 2048,
        'hop_length': 512
    },
    'video': {
        'face_detection_confidence': 0.7
    }
}

# 创建必要的目录
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('training_logs', exist_ok=True)