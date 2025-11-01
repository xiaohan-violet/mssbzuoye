import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

from config import DEVICE, TRAIN_CONFIG
from utils.data_loader import get_data_loaders
from models.text_models import TextModel, TextCNNModel
from models.audio_models import AudioFCN, AudioSimple, TIMNET, CAMPlusPlus
from models.video_models import VideoModel, VideoLSTM
from models.fusion_models import MTFN, MLMF, LateFusion


class Trainer:
    def __init__(self, model, model_name, train_loader, val_loader, test_loader,
                 learning_rate=1e-4, weight_decay=1e-5):
        self.model = model.to(DEVICE)
        self.model_name = model_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.criterion = nn.CrossEntropyLoss()# 交叉熵损失函数
        self.optimizer = optim.Adam(
            model.parameters(),   # 要优化的参数
            lr=learning_rate,     # 学习率
            weight_decay=weight_decay
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )# 动态学习率调整，损失不下降时降低学习率

        self.best_val_acc = 0
        self.train_losses = []
        self.val_accuracies = []
        self.test_accuracy = 0

        # 创建保存目录
        os.makedirs('checkpoints', exist_ok=True)

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f'Training {self.model_name}')

        for batch_idx, batch in enumerate(pbar):
            if batch is None:
                print(f"跳过空批次 {batch_idx}")
                continue

            try:
                # 检查批次数据是否有效
                if 'label' not in batch or batch['label'].nelement() == 0:
                    print(f"批次 {batch_idx} 标签为空")
                    continue

                # 根据模型类型处理输入
                inputs, labels = self._prepare_batch(batch)

                # 检查输入和标签是否有效
                if labels.nelement() == 0:
                    print(f"批次 {batch_idx} 标签无效")
                    continue

                labels = labels.to(DEVICE)
                # 3. 梯度清零
                self.optimizer.zero_grad()

                # 前向传播（根据模型类型调用不同的前向方法）
                if self.model_name in ['text_model', 'text_cnn']:
                    # 检查文本输入是否有效
                    if 'input_ids' not in inputs or 'attention_mask' not in inputs:
                        print(f"批次 {batch_idx} 文本输入无效")
                        continue

                    input_ids = inputs['input_ids'].to(DEVICE)
                    attention_mask = inputs['attention_mask'].to(DEVICE)
                    outputs = self.model(input_ids, attention_mask)


                elif self.model_name in ['audio_fcn', 'audio_simple']:
                    # 直接使用二维音频特征
                    audio_input = inputs.to(DEVICE)
                    outputs = self.model(audio_input)


                elif self.model_name in ['audio_timnet', 'audio_cam']:
                    # 使用适配后的时序模型
                    audio_input = inputs.to(DEVICE)
                    outputs = self.model(audio_input)

                elif self.model_name in ['mtfn', 'mlmf', 'late_fusion']:
                    text_features = inputs['text'].to(DEVICE)
                    audio_features = inputs['audio'].to(DEVICE)
                    video_features = inputs['video'].to(DEVICE)
                    outputs = self.model(text_features, audio_features, video_features)

                else:
                    outputs = self.model(inputs.to(DEVICE))

                # 检查输出是否有效
                if outputs.nelement() == 0:
                    print(f"批次 {batch_idx} 输出为空")
                    continue

                # 计算损失
                loss = self.criterion(outputs, labels)

                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += loss.item()

                # 计算准确率
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # 更新进度条
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100 * correct / total:.2f}%'
                })

            except Exception as e:
                print(f"训练批次 {batch_idx} 错误: {e}")
                continue

        if len(self.train_loader) > 0 and total > 0:
            avg_loss = total_loss / len(self.train_loader)
            accuracy = 100 * correct / total
        else:
            avg_loss = 0
            accuracy = 0

        return avg_loss, accuracy

    def validate(self, loader_type='val'):
        """验证模型"""
        self.model.eval()
        correct = 0
        total = 0

        loader = self.val_loader if loader_type == 'val' else self.test_loader

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                if batch is None:
                    continue

                try:
                    inputs, labels = self._prepare_batch(batch)

                    if labels.nelement() == 0:
                        continue

                    labels = labels.to(DEVICE)

                    # 前向传播
                    if self.model_name in ['text_model', 'text_cnn']:
                        if 'input_ids' not in inputs or 'attention_mask' not in inputs:
                            continue

                        input_ids = inputs['input_ids'].to(DEVICE)
                        attention_mask = inputs['attention_mask'].to(DEVICE)
                        outputs = self.model(input_ids, attention_mask)

                    elif self.model_name in ['audio_timnet', 'audio_cam']:
                        audio_input = inputs.to(DEVICE)
                        if audio_input.dim() == 2:
                            audio_features = audio_input.unsqueeze(1)
                            if audio_features.shape[1] < 50:
                                repeat_times = (50 + audio_features.shape[1] - 1) // audio_features.shape[1]
                                audio_features = audio_features.repeat(1, repeat_times, 1)
                            audio_features = audio_features[:, :50, :]
                        else:
                            audio_features = audio_input
                        outputs = self.model(audio_features)

                    elif self.model_name in ['video_model', 'video_lstm']:
                        video_input = inputs.to(DEVICE)
                        outputs = self.model(video_input)

                    elif self.model_name in ['mtfn', 'mlmf', 'late_fusion']:
                        text_features = inputs['text'].to(DEVICE)
                        audio_features = inputs['audio'].to(DEVICE)
                        video_features = inputs['video'].to(DEVICE)
                        outputs = self.model(text_features, audio_features, video_features)

                    else:
                        outputs = self.model(inputs.to(DEVICE))

                    # 计算准确率
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                except Exception as e:
                    print(f"验证批次 {batch_idx} 错误: {e}")
                    continue

        accuracy = 100 * correct / total if total > 0 else 0
        return accuracy

    # 在 Trainer 类的 _prepare_batch 方法中添加数据类型检查

    def _prepare_batch(self, batch):
        """准备批次数据 - 修复数据类型"""
        labels = batch['label']

        if self.model_name in ['text_model', 'text_cnn']:
            inputs = {
                'input_ids': batch['text_input']['input_ids'].to(DEVICE),
                'attention_mask': batch['text_input']['attention_mask'].to(DEVICE)
            }
        elif self.model_name in ['audio_fcn', 'audio_simple', 'audio_timnet', 'audio_cam']:
            # 确保音频特征是float类型
            inputs = batch['audio'].to(DEVICE).float()
        elif self.model_name in ['video_model', 'video_lstm']:
            # 确保视频特征是float类型
            inputs = batch['video'].to(DEVICE).float()
        elif self.model_name in ['mtfn', 'mlmf', 'late_fusion']:
            # 确保所有融合特征是float类型
            inputs = {
                'text': batch['text_features'].to(DEVICE).float(),
                'audio': batch['audio'].to(DEVICE).float(),
                'video': batch['video'].to(DEVICE).float()
            }
        else:
            inputs = batch['text_input']['input_ids'].to(DEVICE)

        return inputs, labels

    def train(self, epochs=50, patience=10):
        """完整训练过程"""
        print(f"开始训练 {self.model_name}...")

        for epoch in range(epochs):
            start_time = time.time()

            # 训练一个epoch
            train_loss, train_acc = self.train_epoch()

            # 验证
            val_acc = self.validate('val')
            test_acc = self.validate('test')

            # 学习率调度
            if train_loss > 0:
                self.scheduler.step(train_loss)

            # 记录结果
            self.train_losses.append(train_loss)
            self.val_accuracies.append(val_acc)

            epoch_time = time.time() - start_time

            print(f'Epoch {epoch + 1}/{epochs}, Time: {epoch_time:.2f}s')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Acc: {val_acc:.2f}%, Test Acc: {test_acc:.2f}%')
            print('-' * 50)

            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.test_accuracy = test_acc
                torch.save(
                    self.model.state_dict(),
                    f'checkpoints/best_{self.model_name}.pth'
                )
                print(f'新的最佳模型已保存，验证准确率: {val_acc:.2f}%, 测试准确率: {test_acc:.2f}%')

            # 早停
            if epoch > patience and len(self.val_accuracies) > patience:
                recent_accs = self.val_accuracies[-patience:]
                if max(recent_accs) < self.best_val_acc:
                    print(f'早停触发，在第 {epoch + 1} 轮停止训练')
                    break

        print(f'训练完成，最佳验证准确率: {self.best_val_acc:.2f}%, 最终测试准确率: {self.test_accuracy:.2f}%')

        return self.train_losses, self.val_accuracies


def train_all_models():
    """训练所有模型"""
    results = {}

    # 单模态模型
    print("=" * 60)
    print("训练单模态模型...")
    print("=" * 60)

    # 文本模型（保持不变）
    print("\n>>> 训练文本BERT模型...")
    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=TRAIN_CONFIG['batch_size'],
        modality='text'
    )

    text_model = TextModel()
    text_trainer = Trainer(
        text_model, 'text_model', train_loader, val_loader, test_loader,
        learning_rate=TRAIN_CONFIG['learning_rate'],
        weight_decay=TRAIN_CONFIG['weight_decay']
    )
    text_trainer.train(
        epochs=TRAIN_CONFIG['epochs'],
        patience=TRAIN_CONFIG['patience']
    )
    results['text_model'] = text_trainer.test_accuracy

    # 文本CNN模型（保持不变）
    print("\n>>> 训练文本CNN模型...")
    text_cnn_model = TextCNNModel()
    text_cnn_trainer = Trainer(
        text_cnn_model, 'text_cnn', train_loader, val_loader, test_loader,
        learning_rate=TRAIN_CONFIG['learning_rate'],
        weight_decay=TRAIN_CONFIG['weight_decay']
    )
    text_cnn_trainer.train(
        epochs=TRAIN_CONFIG['epochs'],
        patience=TRAIN_CONFIG['patience']
    )
    results['text_cnn'] = text_cnn_trainer.test_accuracy

    # 音频模型 - 使用新的全连接网络
    print("\n>>> 训练音频全连接模型...")
    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=TRAIN_CONFIG['batch_size'],
        modality='audio'
    )

    audio_fcn = AudioFCN()
    audio_fcn_trainer = Trainer(
        audio_fcn, 'audio_fcn', train_loader, val_loader, test_loader,
        learning_rate=TRAIN_CONFIG['learning_rate'],
        weight_decay=TRAIN_CONFIG['weight_decay']
    )
    audio_fcn_trainer.train(
        epochs=TRAIN_CONFIG['epochs'],
        patience=TRAIN_CONFIG['patience']
    )
    results['audio_fcn'] = audio_fcn_trainer.test_accuracy

    # 音频简单模型
    print("\n>>> 训练音频简单模型...")
    audio_simple = AudioSimple()
    audio_simple_trainer = Trainer(
        audio_simple, 'audio_simple', train_loader, val_loader, test_loader,
        learning_rate=TRAIN_CONFIG['learning_rate'],
        weight_decay=TRAIN_CONFIG['weight_decay']
    )
    audio_simple_trainer.train(
        epochs=TRAIN_CONFIG['epochs'],
        patience=TRAIN_CONFIG['patience']
    )
    results['audio_simple'] = audio_simple_trainer.test_accuracy

    # 视频模型（保持不变）
    print("\n>>> 训练视频模型...")
    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=TRAIN_CONFIG['batch_size'],
        modality='video'
    )

    video_model = VideoModel()
    video_trainer = Trainer(
        video_model, 'video_model', train_loader, val_loader, test_loader,
        learning_rate=TRAIN_CONFIG['learning_rate'],
        weight_decay=TRAIN_CONFIG['weight_decay']
    )
    video_trainer.train(
        epochs=TRAIN_CONFIG['epochs'],
        patience=TRAIN_CONFIG['patience']
    )
    results['video_model'] = video_trainer.test_accuracy

    # 多模态融合模型（保持不变）
    print("=" * 60)
    print("训练多模态融合模型...")
    print("=" * 60)

    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=TRAIN_CONFIG['batch_size'],
        modality='all'
    )

    # MTFN模型
    print("\n>>> 训练MTFN多模态融合模型...")
    mtfn_model = MTFN()
    mtfn_trainer = Trainer(
        mtfn_model, 'mtfn', train_loader, val_loader, test_loader,
        learning_rate=TRAIN_CONFIG['learning_rate'],
        weight_decay=TRAIN_CONFIG['weight_decay']
    )
    mtfn_trainer.train(
        epochs=TRAIN_CONFIG['epochs'],
        patience=TRAIN_CONFIG['patience']
    )
    results['mtfn'] = mtfn_trainer.test_accuracy

    # MLMF模型
    print("\n>>> 训练MLMF多模态融合模型...")
    mlmf_model = MLMF()
    mlmf_trainer = Trainer(
        mlmf_model, 'mlmf', train_loader, val_loader, test_loader,
        learning_rate=TRAIN_CONFIG['learning_rate'],
        weight_decay=TRAIN_CONFIG['weight_decay']
    )
    mlmf_trainer.train(
        epochs=TRAIN_CONFIG['epochs'],
        patience=TRAIN_CONFIG['patience']
    )
    results['mlmf'] = mlmf_trainer.test_accuracy

    # Late Fusion模型
    print("\n>>> 训练Late Fusion模型...")
    late_fusion_model = LateFusion()
    late_fusion_trainer = Trainer(
        late_fusion_model, 'late_fusion', train_loader, val_loader, test_loader,
        learning_rate=TRAIN_CONFIG['learning_rate'],
        weight_decay=TRAIN_CONFIG['weight_decay']
    )
    late_fusion_trainer.train(
        epochs=TRAIN_CONFIG['epochs'],
        patience=TRAIN_CONFIG['patience']
    )
    results['late_fusion'] = late_fusion_trainer.test_accuracy

    # 输出最终结果
    print("\n" + "=" * 60)
    print("所有模型训练完成！最终结果:")
    print("=" * 60)
    for model_name, accuracy in results.items():
        print(f"{model_name:15}: {accuracy:.2f}%")

    best_model = max(results.items(), key=lambda x: x[1])
    print(f"\n最佳模型: {best_model[0]}, 测试准确率: {best_model[1]:.2f}%")


if __name__ == '__main__':
    train_all_models()