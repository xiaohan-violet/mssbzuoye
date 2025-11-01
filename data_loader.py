import pandas as pd
import pickle
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import os
import random
import jieba
from transformers import BertTokenizer, BertModel
from config import DATA_PATH, MODEL_CONFIG, DEVICE


class DataAugmentation:
    """数据增强类"""

    def __init__(self):
        # 加载同义词词典（你可以替换为自己的同义词文件）
        self.synonyms_dict = self._load_synonyms()

    def _load_synonyms(self):
        """加载同义词词典"""
        synonyms = {
            '好': ['不错', '良好', '优秀', '出色'],
            '坏': ['糟糕', '差劲', '不好', '恶劣'],
            '喜欢': ['爱', '喜爱', '钟爱', '热爱'],
            '讨厌': ['厌恶', '憎恶', '反感', '不喜欢'],
            '高兴': ['开心', '快乐', '愉快', '兴奋'],
            '悲伤': ['伤心', '难过', '悲哀', '痛苦'],
            # 可以继续添加更多同义词
        }
        return synonyms

    def augment_text(self, text, aug_prob=0.3):
        """文本数据增强"""
        if random.random() > aug_prob or len(text) < 2:
            return text

        try:
            words = list(jieba.cut(text))
            if len(words) < 2:
                return text

            aug_type = random.choice(['synonym', 'delete', 'swap', 'insert'])

            if aug_type == 'synonym' and self.synonyms_dict:
                # 同义词替换
                new_words = words.copy()
                for i, word in enumerate(new_words):
                    if random.random() < 0.3 and word in self.synonyms_dict:
                        synonyms = self.synonyms_dict[word]
                        new_words[i] = random.choice(synonyms)
                return ''.join(new_words)

            elif aug_type == 'delete':
                # 随机删除
                if len(words) > 3:
                    delete_idx = random.randint(0, len(words) - 1)
                    new_words = words[:delete_idx] + words[delete_idx + 1:]
                    return ''.join(new_words)

            elif aug_type == 'swap':
                # 随机交换
                if len(words) > 2:
                    idx1, idx2 = random.sample(range(len(words)), 2)
                    new_words = words.copy()
                    new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
                    return ''.join(new_words)

            elif aug_type == 'insert':
                # 随机插入
                insert_idx = random.randint(0, len(words))
                insert_word = random.choice(['很', '非常', '特别', '真的'])
                new_words = words[:insert_idx] + [insert_word] + words[insert_idx:]
                return ''.join(new_words)

        except Exception as e:
            print(f"文本增强错误: {e}")

        return text

    def augment_audio(self, audio_features, aug_prob=0.3):
        """音频特征增强"""
        if random.random() > aug_prob:
            return audio_features

        try:
            aug_type = random.choice(['noise', 'scale', 'shift'])

            if aug_type == 'noise':
                # 添加高斯噪声
                noise = torch.randn_like(audio_features) * 0.01
                return audio_features + noise

            elif aug_type == 'scale':
                # 随机缩放
                scale = random.uniform(0.9, 1.1)
                return audio_features * scale

            elif aug_type == 'shift':
                # 随机平移
                shift = random.uniform(-0.1, 0.1)
                return audio_features + shift

        except Exception as e:
            print(f"音频增强错误: {e}")

        return audio_features

    def augment_video(self, video_features, aug_prob=0.3):
        """视频特征增强"""
        if random.random() > aug_prob:
            return video_features

        try:
            aug_type = random.choice(['noise', 'scale', 'dropout'])

            if aug_type == 'noise':
                # 添加高斯噪声
                noise = torch.randn_like(video_features) * 0.01
                return video_features + noise

            elif aug_type == 'scale':
                # 随机缩放
                scale = random.uniform(0.9, 1.1)
                return video_features * scale

            elif aug_type == 'dropout':
                # 随机丢弃部分特征
                mask = torch.rand_like(video_features) > 0.1
                return video_features * mask.float()

        except Exception as e:
            print(f"视频增强错误: {e}")

        return video_features


# 在 CHSIMSv2Dataset 类的 __init__ 方法中，修改数据分割逻辑：
class CHSIMSv2Dataset(Dataset):
    def __init__(self, metadata_path, features_path, mode='train', modality='all', augment=False):
        """
        CH-SIMS v2数据集加载器 - 优化版本
        """
        self.mode = mode
        self.modality = modality
        self.augment = augment and mode == 'train'  # 只在训练时增强

        # 初始化数据增强
        if self.augment:
            self.augmentor = DataAugmentation()

        # 加载元数据
        self.metadata = pd.read_csv(metadata_path)
        print(f"元数据加载成功: {len(self.metadata)} 个样本")
        print(f"元数据列名: {self.metadata.columns.tolist()}")

        # 计算类别权重
        self.class_weights = self._calculate_class_weights()

        # 加载特征数据
        with open(features_path, 'rb') as f:
            self.features_data = pickle.load(f)
        print(f"特征数据加载成功，包含: {list(self.features_data.keys())}")

        # 根据模式选择数据分割 - 修复：使用正确的列名
        if mode == 'train':
            self.split_data = self.features_data['train']
            # 过滤元数据 - 使用正确的列名 'mode'
            self.metadata = self.metadata[self.metadata['mode'] == 'train']
        elif mode == 'valid':
            self.split_data = self.features_data['valid']
            self.metadata = self.metadata[self.metadata['mode'] == 'test']  # 注意：验证集可能也用test
        else:  # test
            self.split_data = self.features_data['test']
            self.metadata = self.metadata[self.metadata['mode'] == 'test']

        # 初始化BERT tokenizer和模型
        print("初始化BERT模型...")
        self.tokenizer = BertTokenizer.from_pretrained(MODEL_CONFIG['text']['bert_model'])
        self.bert_model = BertModel.from_pretrained(MODEL_CONFIG['text']['bert_model']).to(DEVICE)
        self.bert_model.eval()

        # 预计算所有文本特征，避免在getitem中重复计算
        self.text_features_cache = {}
        print("预计算文本特征...")
        for idx in range(len(self.split_data['id'])):
            text = str(self.split_data['raw_text'][idx])
            with torch.no_grad():
                inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                outputs = self.bert_model(**inputs)
                text_features = outputs.last_hidden_state[:, 0, :].cpu().squeeze(0)
                self.text_features_cache[idx] = text_features

        print(f"加载 {mode} 数据集: {len(self.split_data['id'])} 个样本")
        print(f"类别分布: {self._get_class_distribution()}")

    def _calculate_class_weights(self):
        """计算类别权重用于处理不平衡数据"""
        class_counts = {0: 0, 1: 0, 2: 0}

        for idx in range(len(self.metadata)):
            regression_label = float(self.metadata.iloc[idx]['regression_labels'])
            label_class = self._continuous_to_class(regression_label)
            class_counts[label_class] += 1

        total_samples = sum(class_counts.values())
        class_weights = {
            class_id: total_samples / count if count > 0 else 1.0
            for class_id, count in class_counts.items()
        }

        # 归一化权重
        weight_sum = sum(class_weights.values())
        class_weights = {k: v / weight_sum for k, v in class_weights.items()}

        print(f"类别权重: {class_weights}")
        return class_weights

    def _get_class_distribution(self):
        """获取类别分布"""
        class_counts = {0: 0, 1: 0, 2: 0}
        for idx in range(len(self.split_data['id'])):
            regression_label = float(self.split_data['regression_labels'][idx])
            label_class = self._continuous_to_class(regression_label)
            class_counts[label_class] += 1
        return class_counts

    def get_sample_weights(self):
        """获取每个样本的权重用于采样"""
        sample_weights = []
        for idx in range(len(self.split_data['id'])):
            regression_label = float(self.split_data['regression_labels'][idx])
            label_class = self._continuous_to_class(regression_label)
            sample_weights.append(self.class_weights[label_class])
        return sample_weights

    def __len__(self):
        return len(self.split_data['id'])

    def __getitem__(self, idx):
        try:
            # 获取样本ID - 确保是字符串
            sample_id = str(self.split_data['id'][idx])

            # 获取标签 - 使用回归标签并转换为分类
            regression_label = float(self.split_data['regression_labels'][idx])
            label_class = self._continuous_to_class(regression_label)

            # 文本特征和输入
            text = str(self.split_data['raw_text'][idx])

            # 数据增强 - 文本
            if self.augment:
                text = self.augmentor.augment_text(text)

            text_input = self._process_text(text)
            text_features = self.text_features_cache[idx]

            # 音频特征
            audio_features = self._get_audio_features(idx)
            if self.augment:
                audio_features = self.augmentor.augment_audio(audio_features)

            # 视频特征
            video_features = self._get_video_features(idx)
            if self.augment:
                video_features = self.augmentor.augment_video(video_features)

            sample = {
                'text_features': text_features,
                'text_input': text_input,
                'audio': audio_features,
                'video': video_features,
                'label': label_class,
                'label_continuous': torch.tensor(regression_label, dtype=torch.float32),
                'sample_id': sample_id
            }

            return sample

        except Exception as e:
            print(f"处理样本 {idx} 时出错: {e}")
            return self._create_empty_sample()

    def _get_audio_features(self, idx):
        """获取音频特征"""
        if idx < len(self.split_data['audio']):
            audio_data = self.split_data['audio'][idx]
            if isinstance(audio_data, np.ndarray):
                # 如果音频特征是多维的，取均值
                if audio_data.ndim > 1:
                    audio_features = np.mean(audio_data, axis=0)
                else:
                    audio_features = audio_data

                # 转换为tensor并确保维度正确
                audio_tensor = torch.tensor(audio_features, dtype=torch.float32)

                # 调整到目标维度
                if audio_tensor.shape[0] > MODEL_CONFIG['audio']['input_dim']:
                    audio_tensor = audio_tensor[:MODEL_CONFIG['audio']['input_dim']]
                elif audio_tensor.shape[0] < MODEL_CONFIG['audio']['input_dim']:
                    padding = torch.zeros(MODEL_CONFIG['audio']['input_dim'] - audio_tensor.shape[0])
                    audio_tensor = torch.cat([audio_tensor, padding])

                return audio_tensor

        # 默认返回零向量
        return torch.zeros(MODEL_CONFIG['audio']['input_dim'], dtype=torch.float32)

    def _get_video_features(self, idx):
        """获取视频特征"""
        if idx < len(self.split_data['vision']):
            vision_data = self.split_data['vision'][idx]
            if isinstance(vision_data, np.ndarray):
                # 如果视觉特征是多维的，取均值
                if vision_data.ndim > 1:
                    video_features = np.mean(vision_data, axis=0)
                else:
                    video_features = vision_data

                # 转换为tensor并确保维度正确
                video_tensor = torch.tensor(video_features, dtype=torch.float32)

                # 调整到目标维度
                if video_tensor.shape[0] > MODEL_CONFIG['video']['input_dim']:
                    video_tensor = video_tensor[:MODEL_CONFIG['video']['input_dim']]
                elif video_tensor.shape[0] < MODEL_CONFIG['video']['input_dim']:
                    padding = torch.zeros(MODEL_CONFIG['video']['input_dim'] - video_tensor.shape[0])
                    video_tensor = torch.cat([video_tensor, padding])

                return video_tensor

        # 默认返回零向量
        return torch.zeros(MODEL_CONFIG['video']['input_dim'], dtype=torch.float32)

    def _continuous_to_class(self, label):
        """将连续标签转换为分类标签"""
        # 原始标签是-1到1的连续值，转换成3分类
        if label <= -0.2:
            return 0  # 负面
        elif label >= 0.2:
            return 2  # 正面
        else:
            return 1  # 中性

    def _process_text(self, text):
        """处理文本数据用于BERT模型"""
        encoded = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=MODEL_CONFIG['text']['max_length'],
            return_tensors='pt'
        )

        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0)
        }

    def _create_empty_sample(self):
        """创建空样本"""
        text = "空文本"
        encoded = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=MODEL_CONFIG['text']['max_length'],
            return_tensors='pt'
        )

        return {
            'text_features': torch.zeros(768, dtype=torch.float32),
            'text_input': {
                'input_ids': encoded['input_ids'].squeeze(0),
                'attention_mask': encoded['attention_mask'].squeeze(0)
            },
            'audio': torch.zeros(MODEL_CONFIG['audio']['input_dim'], dtype=torch.float32),
            'video': torch.zeros(MODEL_CONFIG['video']['input_dim'], dtype=torch.float32),
            'label': 1,  # 中性
            'label_continuous': torch.tensor(0.0, dtype=torch.float32),
            'sample_id': 'error'
        }


def custom_collate_fn(batch):
    """自定义collate函数处理数据类型问题"""
    # 过滤掉None和无效样本
    batch = [item for item in batch if item is not None]

    if len(batch) == 0:
        # 返回一个最小的有效批次
        return _create_minimal_batch()

    # 手动处理每个字段
    collated_batch = {}

    for key in batch[0].keys():
        if key == 'text_input':
            # 特殊处理text_input字典
            collated_batch[key] = {
                'input_ids': torch.stack([item[key]['input_ids'] for item in batch]).long(),
                'attention_mask': torch.stack([item[key]['attention_mask'] for item in batch]).long()
            }
        elif key == 'sample_id':
            # sample_id保持为列表
            collated_batch[key] = [item[key] for item in batch]
        elif key == 'label':
            # label转换为long tensor
            collated_batch[key] = torch.tensor([item[key] for item in batch], dtype=torch.long)
        else:
            # 其他字段直接stack，确保是float类型
            try:
                # 确保所有元素都是float tensor
                tensor_list = []
                for item in batch:
                    tensor_item = item[key]
                    if not isinstance(tensor_item, torch.Tensor):
                        tensor_item = torch.tensor(tensor_item)
                    # 转换为float类型
                    tensor_list.append(tensor_item.float())
                collated_batch[key] = torch.stack(tensor_list)
            except Exception as e:
                print(f"处理字段 {key} 时出错: {e}")
                # 如果stack失败，创建默认值
                if key == 'text_features':
                    collated_batch[key] = torch.zeros(len(batch), 768, dtype=torch.float32)
                elif key == 'audio':
                    collated_batch[key] = torch.zeros(len(batch), MODEL_CONFIG['audio']['input_dim'],
                                                      dtype=torch.float32)
                elif key == 'video':
                    collated_batch[key] = torch.zeros(len(batch), MODEL_CONFIG['video']['input_dim'],
                                                      dtype=torch.float32)
                elif key == 'label_continuous':
                    collated_batch[key] = torch.zeros(len(batch), dtype=torch.float32)

    return collated_batch


def _create_minimal_batch():
    """创建最小有效批次"""
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(MODEL_CONFIG['text']['bert_model'])

    text = "最小批次文本"
    encoded = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=MODEL_CONFIG['text']['max_length'],
        return_tensors='pt'
    )

    return {
        'text_features': torch.zeros(1, 768, dtype=torch.float32),
        'text_input': {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        },
        'audio': torch.zeros(1, MODEL_CONFIG['audio']['input_dim'], dtype=torch.float32),
        'video': torch.zeros(1, MODEL_CONFIG['video']['input_dim'], dtype=torch.float32),
        'label': torch.tensor([1], dtype=torch.long),
        'label_continuous': torch.tensor([0.0], dtype=torch.float32),
        'sample_id': ['minimal_batch']
    }


def get_data_loaders(batch_size=32, modality='all', use_augmentation=True, use_class_balancing=True):
    """获取训练和测试数据加载器 - 优化版本"""

    metadata_path = DATA_PATH['metadata']
    features_path = DATA_PATH['unaligned']

    print(f"加载数据...")
    print(f"元数据路径: {metadata_path}")
    print(f"特征路径: {features_path}")
    print(f"使用数据增强: {use_augmentation}")
    print(f"使用类别平衡: {use_class_balancing}")

    # 训练数据集
    train_dataset = CHSIMSv2Dataset(
        metadata_path,
        features_path,
        mode='train',
        modality=modality,
        augment=use_augmentation
    )

    # 验证数据集
    valid_dataset = CHSIMSv2Dataset(
        metadata_path,
        features_path,
        mode='valid',
        modality=modality,
        augment=False  # 验证集不使用增强
    )

    # 测试数据集
    test_dataset = CHSIMSv2Dataset(
        metadata_path,
        features_path,
        mode='test',
        modality=modality,
        augment=False  # 测试集不使用增强
    )

    # 类别平衡采样器
    train_sampler = None
    if use_class_balancing:
        sample_weights = train_dataset.get_sample_weights()
        train_sampler = WeightedRandomSampler(
            sample_weights,
            len(sample_weights),
            replacement=True
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),  # 如果使用sampler，shuffle应为False
        sampler=train_sampler,
        collate_fn=custom_collate_fn
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn
    )

    print(f"数据加载完成:")
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(valid_dataset)} 样本")
    print(f"测试集: {len(test_dataset)} 样本")

    if use_class_balancing:
        print("使用加权随机采样器平衡类别分布")

    return train_loader, valid_loader, test_loader